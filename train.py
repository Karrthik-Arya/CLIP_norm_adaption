# from lavis.models import load_model_and_preprocess
from PIL import Image
import requests
import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class TransferModel(nn.Module):
    def __init__(self,num_classes=1000, caption_embedding_dim=128):
        super(TransferModel, self).__init__()
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
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
        self.caption_projectors = [nn.Linear(caption_embedding_dim, text_hidden_size) for i in range(3)]
        # self.total_layers = len(self.model.transformer.resblocks)
        # self.layers_to_modify = sorted(random.sample(range(self.total_layers), 3))
        self.layers_to_modify = [3, 6, 9]

        self.classifier = nn.Linear(1024,num_classes)

    def embed_caption(self, caption):
        """Embed the caption into the same dimensionality"""
        tokens = clip.tokenize(caption).to(self.device)
        with torch.no_grad():
            caption_embeddings = self.model.encode_text(tokens)
            caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)
        return caption_embeddings


    def forward(self, image, text, caption_embeddings):
        if "source" in text:
            inputs1 = clip.tokenize(text["source"]).to(self.device)
            image_features1 = self.source_model.encode_image(image["source"].to(self.device))

            text_encoder1 = self.source_model.transformer
            text_embedding1 = self.source_model.token_embedding(inputs1)
            position_embedding1 = self.source_model.positional_embedding[: text_embedding1.size(1), :]

            
            x = text_embedding1 + position_embedding1
            x = x.permute(1, 0, 2)
            caption_idx = 0
            # text_features1 = self.source_model.encode_text(inputs1)

            for i, layer in enumerate(text_encoder1.resblocks):
                x = layer(x)
                if i in self.layers_to_modify:
                    caption_embeddings_projected = self.caption_projectors[caption_idx](caption_embeddings).unsqueeze(0)
                    x = torch.cat([x, caption_embeddings_projected], dim=0)
                    caption_idx += 1

            x = x.permute(1, 0, 2)
            x = self.source_model.ln_final(x)  
            text_features1 = x[torch.arange(x.shape[0]), inputs1.argmax(dim=-1)]  
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)

        if "target" in text:
            inputs2 = clip.tokenize(text["target"]).to(self.device)
            image_features2 = self.target_model.encode_image(image["target"].to(self.device))
            text_encoder2 = self.target_model.transformer
            text_embedding2 = self.target_model.token_embedding(inputs2)
            position_embedding2 = self.target_model.positional_embedding[: text_embedding2.size(1), :]

            
            x = text_embedding2 + position_embedding2
            x = x.permute(1, 0, 2)
            # text_features1 = self.source_model.encode_text(inputs1)
            caption_idx = 0

            for i, layer in enumerate(text_encoder2.resblocks):
                x = layer(x)
                if i in self.layers_to_modify:
                    caption_embeddings_projected = self.caption_projectors[caption_idx](caption_embeddings).unsqueeze(0)
                    x = torch.cat([x, caption_embeddings_projected], dim=0)
                    caption_idx += 1

            x = x.permute(1, 0, 2)
            x = self.target_model.ln_final(x)  
            text_features2 = x[torch.arange(x.shape[0]), inputs2.argmax(dim=-1)]  
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        
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
    
batch_size=128
num_workers=4
lr =1e-3
epochs = 50
momentum = 0.99
image_size = 224




mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_dataset = TrainDataset('data/vqa_v2','train')
val_dataset = TrainDataset('data/vqa_v2','val','VQAv2')
train_targ_dataset = TestDataset('data/test/images', 'data/test/train_questions.csv')
test_targ_dataset = TestDataset('data/test/images', 'data/test/test_questions.csv')

train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=int(batch_size*0.75), shuffle=False)
train_targ_loader = DataLoader(train_targ_dataset, num_workers=num_workers, batch_size=int(batch_size*0.25), shuffle=False)
val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
# cross_loader = DataLoader(cross_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

train_loader_itr = iter(train_loader)
train_targ_loader_itr = iter(train_targ_loader)

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

transfer_model = transfer_model.to('cuda:1')
optimizer = AdamW(transfer_model.parameters(), lr=lr)

optimizer.zero_grad()

train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
# cross_loss_meter = AverageMeter()
train_accuracy_meter = AverageMeter()
val_accuracy_meter = AverageMeter()
# cross_accuracy_meter = AverageMeter()
best_val_acc = 0

for i in range(epochs):

    transfer_model.train()
    train_loss_meter.reset()
    train_accuracy_meter.reset()
    mixed_loader = mixed_data_loader(train_loader_itr, train_targ_loader_itr)

    for data in tqdm(mixed_loader):
        img = data["img"]
        ques = data["question"]
        ans = data["answer"].to('cuda:1')
        captions = data["caption"]
        # captions["source"] = [transfer_model.generate_caption(image) for image in img["source"]]
        # captions["target"] = [transfer_model.generate_caption(image) for image in img["target"]]
        
        captions["source"] = [transfer_model.embed_caption(caption) for caption in captions["source"]]
        captions["target"] = [transfer_model.embed_caption(caption) for caption in captions["target"]]

        output = transfer_model(img, ques, captions)
        # print(output.shape)
        # print(ans.shape)
        # print(ans)
        cosine_sim = F.cosine_similarity(ln_params['source'], ln_params['target'], dim=0)
        cosine_loss = -cosine_sim.mean()

        loss =  torch.nn.CrossEntropyLoss()(output[:len(ques['source'])],ans[:len(ques['source'])])
        probs = F.softmax(output[len(ques['source']):], dim=-1)
        self_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        loss +=  self_entropy
        loss += cosine_loss
            
        train_loss_meter.update(loss.item(), ans.size(0))
        # Calculate and update accuracy
        acc1 = accuracy(output, ans, topk=(1,))
        train_accuracy_meter.update(acc1[0].item(), ans.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # break
    print(f'Epoch: {i+1}, Training Loss: {train_loss_meter.avg:.4f}, Training Accuracy: {train_accuracy_meter.avg:.2f} ')
    transfer_model.eval()
    val_loss_meter.reset()
    val_accuracy_meter.reset()
    for data in tqdm(val_loader):
        img = data["img"]
        ques = data["question"]
        ans = data["answer"].to('cuda:1')

        img =  {"source": img}
        ques = {"source": ques}

        output = transfer_model(img,ques)
        loss =  torch.nn.CrossEntropyLoss()(output,ans)
        val_loss_meter.update(loss.item(), ans.size(0))
        # Calculate and update validation accuracy
        acc1 = accuracy(output, ans, topk=(1,))
        val_accuracy_meter.update(acc1[0].item(), ans.size(0))
        # break
    # cross_loss_meter.reset()
    # cross_accuracy_meter.reset()
    # for data in tqdm(cross_loader):
    #     img = data["img"]
    #     ques = data["question"]
    #     ans = data["answer"]
    #     img,ans = img.to('cuda'),ans.to('cuda')

    #     output = transfer_model(img,ques)

    #     loss =  torch.nn.CrossEntropyLoss()(output,ans)
    #     # cross_loss_meter.update(loss.item(), img.size(0))
    #     # Calculate and update accuracy
    #     acc1 = accuracy(output, ans, topk=(1,))
        # cross_accuracy_meter.update(acc1[0].item(), img.size(0))
        # break
    # print(val_accuracy_meter.avg)
    # print(val_loss_meter.avg)
    print(f'Epoch: {i+1}, Validation Loss: {val_loss_meter.avg:.4f}, Validation Accuracy: {val_accuracy_meter.avg:.2f} ')
    if best_val_acc< val_accuracy_meter.avg:
        torch.save(transfer_model.state_dict(), './clip_vqa_v2.pth')
        best_val_acc = val_accuracy_meter.avg
        print("Model Saved!!!")

