# from lavis.models import load_model_and_preprocess
from PIL import Image
import requests
import torch
# import torch. multiprocessing as mp
import torch.nn as nn
import clip
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim import SGD,AdamW
import torch.nn.functional as F
from datasets.trainDataset import TrainDataset
from datasets.testDatset import TestDataset
from tqdm import tqdm
import copy
import random
# from transformers import OFATokenizer, OFAModel
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

ln_params = {}

def source_hook(module: nn.LayerNorm, input, output):
    ln_params["source"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

def target_hook(module, input, output):
    ln_params["target"] = torch.cat((module.weight.data.clone(), module.bias.data.clone()), dim=0)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Add layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x shape: [seq_len, batch_size, dim]
        # context shape: [batch_size, dim]
        seq_len, batch_size, dim = x.shape
        
        # First layer norm
        x_norm = self.norm1(x)
        
        # Reshape context to [batch_size, 1, dim]
        if len(context.shape) == 2:
            context = context.unsqueeze(1)
        
        # Project and reshape queries
        q = self.q(x_norm).permute(1, 0, 2)  # [batch_size, seq_len, dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Project and reshape keys
        k = self.k(context)  # [batch_size, 1, dim]
        k = k.reshape(batch_size, 1, self.num_heads, dim // self.num_heads)
        k = k.permute(0, 2, 3, 1)  # [batch_size, num_heads, head_dim, 1]
        
        # Project and reshape values
        v = self.v(context)  # [batch_size, 1, dim]
        v = v.reshape(batch_size, 1, self.num_heads, dim // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, 1, head_dim]

        # Compute attention scores
        attn = (q @ k) * self.scale  # [batch_size, num_heads, seq_len, 1]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        attn_out = (attn @ v).transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        attn_out = attn_out.reshape(batch_size, seq_len, dim)  # [batch_size, seq_len, dim]
        attn_out = attn_out.permute(1, 0, 2)  # [seq_len, batch_size, dim]
        
        # Final projection and residual connection
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        
        # Second layer norm and residual connection
        out = self.norm2(x + attn_out)
        
        return out

class TransferModel(nn.Module):
    def __init__(self,num_classes=1000, caption_embedding_dim=512, use_captions=True):
        super(TransferModel, self).__init__()
        # self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.use_captions = use_captions
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.float()

        self.source_ln = copy.deepcopy(self.model.ln_final)
        self.target_ln = copy.deepcopy(self.model.ln_final)

        self.source_ln.register_forward_hook(source_hook)
        self.target_ln.register_forward_hook(target_hook)

        for param in self.model.parameters():
            param.requires_grad = False

        self.source_ln.requires_grad_ = True
        self.target_ln.requires_grad_ = True

        self.source_model = copy.deepcopy(self.model)
        self.source_model.ln_final = self.source_ln

        self.target_model = copy.deepcopy(self.model)
        self.target_model.ln_final = self.target_ln

        text_hidden_size = self.model.text_projection.shape[1]
        self.orig_seq_length = 77
        
        # Replace caption projectors with cross-attention modules
        self.cross_attentions = nn.ModuleList([
            CrossAttention(dim=text_hidden_size, num_heads=8) for _ in range(3)
        ])

        self.layers_to_modify = [3, 6, 9]
        self.classifier = nn.Linear(1024, num_classes)

    def embed_caption(self, caption):
        """Embed the caption into the same dimensionality"""
        tokens = clip.tokenize(caption).to(device)
        with torch.no_grad():
            caption_embeddings = self.model.encode_text(tokens)
            caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)
        return caption_embeddings.to(device)

    def forward(self, image, text, caption_embeddings):
        if "source" in text:
            inputs1 = clip.tokenize(text["source"]).to(device)
            image_features1 = self.source_model.encode_image(image["source"].to(device))
            if(self.use_captions):
                text_encoder1 = self.source_model.transformer
                text_embedding1 = self.source_model.token_embedding(inputs1)
                position_embedding1 = self.source_model.positional_embedding[: text_embedding1.size(1), :]
                
                x = text_embedding1 + position_embedding1
                x = x.permute(1, 0, 2)
                caption_idx = 0

                for i, layer in enumerate(text_encoder1.resblocks):
                    x = layer(x)
                    if i in self.layers_to_modify:
                        # caption_embeddings["source"] shape is [B, D]
                        x = self.cross_attentions[caption_idx](x, caption_embeddings["source"])
                        caption_idx += 1

                x = x.permute(1, 0, 2)
                x = self.source_model.ln_final(x)  
                text_features1 = x[torch.arange(x.shape[0]), inputs1.argmax(dim=-1)]
                text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            else:
                text_features1 = self.source_model.encode_text(inputs1)

        if "target" in text:
            inputs2 = clip.tokenize(text["target"]).to(device)
            image_features2 = self.target_model.encode_image(image["target"].to(device))
            if(self.use_captions):
                text_encoder2 = self.target_model.transformer
                text_embedding2 = self.target_model.token_embedding(inputs2)
                position_embedding2 = self.target_model.positional_embedding[: text_embedding2.size(1), :]
                
                x = text_embedding2 + position_embedding2
                x = x.permute(1, 0, 2)
                caption_idx = 0

                for i, layer in enumerate(text_encoder2.resblocks):
                    x = layer(x)
                    if i in self.layers_to_modify:
                        # caption_embeddings["target"] shape is [B, D]
                        x = self.cross_attentions[caption_idx](x, caption_embeddings["target"])
                        caption_idx += 1

                x = x.permute(1, 0, 2)
                x = self.target_model.ln_final(x)  
                text_features2 = x[torch.arange(x.shape[0]), inputs2.argmax(dim=-1)]  
                text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            else:
                text_features2 = self.target_model.encode_text(inputs2)
        
        if ("source" in text) and ("target" in text):
            image_features = torch.cat((image_features1, image_features2), dim=0)
            text_features = torch.cat((text_features1, text_features2), dim=0)
        elif "source" in text:
            image_features = image_features1
            text_features = text_features1
        elif "target" in text:
            image_features = image_features2
            text_features = text_features2

        assert torch.isfinite(image_features).all(), "NaN in image features"
        assert torch.isfinite(text_features).all(), "NaN in text features"

        # print(multimodal_emb.shape)
        multi_modal = torch.cat((image_features,text_features),dim=1)
        # print(multi_modal.dtype)
        # print(self.classifier.weight.dtype)

        out = self.classifier(multi_modal)
        return out

def main():
    batch_size=128
    num_workers=4
    base_lr = 1e-4  # Reduced base learning rate
    epochs = 20
    momentum = 0.99
    image_size = 224
    warmup_epochs = 2
    max_grad_norm = 1.0  # For gradient clipping

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = TrainDataset('data/vqa_v2','train')
    val_dataset = TrainDataset('data/vqa_v2','val')
    train_targ_dataset = TestDataset('data/test/images', 'data/test/train_questions.csv', 'data/test/captions.csv')
    test_targ_dataset = TestDataset('data/test/images', 'data/test/test_questions.csv', 'data/test/captions.csv')

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=int(batch_size*0.75), shuffle=False)
    train_targ_loader = DataLoader(train_targ_dataset, num_workers=num_workers, batch_size=int(batch_size*0.25), shuffle=False)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    # cross_loader = DataLoader(cross_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    def mixed_data_loader(loader1, loader2):
        while True:
            try:
                batch1 = next(loader1)
            except StopIteration:
                break
            
            try:
                batch2 = next(loader2)
            except StopIteration:
                break

            mixed_batch = {
                "img": {"source": batch1["img"], "target": batch2["img"]},
                "question": {"source": batch1["question"], "target":batch2["question"]},  
                "answer": torch.cat((batch1["answer"], batch2["answer"]), dim=0),   
                "caption": {"source": batch1["caption"], "target": batch2["caption"]}       
            }
            
            yield mixed_batch

    transfer_model = TransferModel()
    # transfer_model.load_state_dict(torch.load('./captions_clip.pth', map_location=device))
    transfer_model = transfer_model.to(device)

    for param in transfer_model.parameters():
        param.requires_grad = False
    for caption_projector in transfer_model.cross_attentions:
        for param in caption_projector.parameters():
            param.requires_grad = True
    
    for param in transfer_model.classifier.parameters():
        param.requires_grad = True
            
    optimizer = AdamW(transfer_model.parameters(), lr=base_lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    def get_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        return base_lr

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    val_accuracy_meter = AverageMeter()
    best_val_acc = 0

    for i in range(epochs):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(i)

        transfer_model.train()
        train_loss_meter.reset()
        train_accuracy_meter.reset()

        train_loader_itr = iter(train_loader)
        train_targ_loader_itr = iter(train_targ_loader)
        
        mixed_loader = mixed_data_loader(train_loader_itr, train_targ_loader_itr)

        for data in tqdm(mixed_loader):
            img = data["img"]
            ques = data["question"]
            ans = data["answer"].to(device)
            captions = data["caption"]
            
            captions["source"] = torch.stack([transfer_model.embed_caption(caption) for caption in captions["source"]]).to(device)
            captions["target"] = torch.stack([transfer_model.embed_caption(caption) for caption in captions["target"]]).to(device)

            output = transfer_model(img, ques, captions)
            
            cosine_sim = F.cosine_similarity(ln_params['source'], ln_params['target'], dim=0)
            cosine_loss = -cosine_sim.mean()

            loss = torch.nn.CrossEntropyLoss()(output[:len(ques['source'])], ans[:len(ques['source'])])
            probs = F.softmax(output[len(ques['source']):], dim=-1)
            self_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
            loss += self_entropy
            loss += cosine_loss
                
            train_loss_meter.update(loss.item(), ans.size(0))
            acc1 = accuracy(output, ans, topk=(1,))
            train_accuracy_meter.update(acc1[0].item(), ans.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(transfer_model.parameters(), max_grad_norm)
            
            optimizer.step()

        print(f'Epoch: {i+1}, Training Loss: {train_loss_meter.avg:.4f}, Training Accuracy: {train_accuracy_meter.avg:.2f} ')
        
        # Validation
        transfer_model.eval()
        val_loss_meter.reset()
        val_accuracy_meter.reset()
        
        with torch.no_grad():
            for data in tqdm(val_loader):
                img = data["img"]
                ques = data["question"]
                ans = data["answer"].to(device)
                captions = {"source": data["caption"]}
                captions["source"] = torch.stack([transfer_model.embed_caption(caption) for caption in captions["source"]]).to(device)

                img = {"source": img}
                ques = {"source": ques}

                output = transfer_model(img, ques, captions)
                loss = torch.nn.CrossEntropyLoss()(output, ans)
                val_loss_meter.update(loss.item(), ans.size(0))
                acc1 = accuracy(output, ans, topk=(1,))
                val_accuracy_meter.update(acc1[0].item(), ans.size(0))

        print(f'Epoch: {i+1}, Validation Loss: {val_loss_meter.avg:.4f}, Validation Accuracy: {val_accuracy_meter.avg:.2f} ')
        
        if best_val_acc < val_accuracy_meter.avg:
            torch.save(transfer_model.state_dict(), './captions_clip1.pth')
            best_val_acc = val_accuracy_meter.avg
            print("Model Saved!!!")

if __name__ == "__main__": 
    # mp.set_start_method("spawn", force=True)
    main()
