import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from datasets.testDatset import TestDataset
from datasets.trainDataset import TrainDataset
from tqdm import tqdm
import copy

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
        self.caption_projectors = nn.ModuleList(
            [nn.Linear(caption_embedding_dim, text_hidden_size) for i in range(3)]
        )
        self.token_projectors_seq = nn.ModuleList(
            [nn.Linear(self.orig_seq_length + 1, self.orig_seq_length) for _ in range(3)]
        )

        # self.total_layers = len(self.model.transformer.resblocks)
        # self.layers_to_modify = sorted(random.sample(range(self.total_layers), 3))
        self.layers_to_modify = [3, 6, 9]

        self.classifier = nn.Linear(1024,num_classes)

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
                # text_features1 = self.source_model.encode_text(inputs1)

                for i, layer in enumerate(text_encoder1.resblocks):
                    x = layer(x)
                    if i in self.layers_to_modify:
                        proj_caption = self.caption_projectors[caption_idx](caption_embeddings["source"].squeeze(1))
                        proj_caption = proj_caption.unsqueeze(0)
                        y = torch.cat([x, proj_caption], dim=0)
                        y = y.permute(1, 2, 0)
                        y = self.token_projectors_seq[caption_idx](y)
                        y = y.permute(2, 0, 1)
                        x = y
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
                # text_features1 = self.source_model.encode_text(inputs1)
                caption_idx = 0

                for i, layer in enumerate(text_encoder2.resblocks):
                    x = layer(x)
                    if i in self.layers_to_modify:
                        proj_caption = self.caption_projectors[caption_idx](caption_embeddings["target"].squeeze(1))
                        proj_caption = proj_caption.unsqueeze(0)
                        y = torch.cat([x, proj_caption], dim=0)
                        y = y.permute(1, 2, 0)
                        y = self.token_projectors_seq[caption_idx](y)
                        y = y.permute(2, 0, 1)
                        x = y
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
    

transfer_model = TransferModel()
transfer_model = transfer_model.to(device)
state_dict = torch.load('./captions_clip.pth', map_location=device)
transfer_model.load_state_dict(state_dict)
transfer_model.eval()   

test_dataset = TestDataset('data/test/images', 'data/test/test_questions.csv', 'data/test/captions.csv')

batch_size=128
num_workers=4

test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

test_accuracy_meter = AverageMeter()
cat_accuracy_meters = {
    "VQAabs": AverageMeter(),
    "VG": AverageMeter(),
    "GQA": AverageMeter()
}

for data in tqdm(test_loader):
    img = data["img"]
    ques = data["question"]
    ans = data["answer"]
    img_path = data["img_path"]
    img =  {"target": img}
    ques = {"target": ques}

    captions = {"target": torch.stack([transfer_model.embed_caption(caption) for caption in data["caption"]]).to(device)}

    output = transfer_model(img,ques, captions)
    cat_output={
        "GQA": [],
        "VG": [],
        "VQAabs": []
    }
    cat_ans={
        "GQA": [],
        "VG": [],
        "VQAabs": []
    }

    for i, image in enumerate(img_path):
        cat = image.split("_")[0]
        cat_output[cat].append(output[i])
        cat_ans[cat].append(ans[i])



    # loss =  torch.nn.CrossEntropyLoss()(output,ans)
    # val_loss_meter.update(loss.item(), img.size(0))
    # Calculate and update validation accuracy
    for cat in cat_output:
        answer = torch.tensor(cat_ans[cat]).to("cuda:1")
        if(answer.size(0) > 0):
            pred = torch.stack(cat_output[cat]).to("cuda:1")
            acc1 = accuracy(pred, answer, topk=(1,))
            cat_accuracy_meters[cat].update(acc1[0].item(), answer.size(0))
    # print(cat_accuracy_meters["VQAabs"].avg)

    ans = ans.to("cuda:1")
    acc1 = accuracy(output, ans, topk=(1,))
    # print(acc1)
    test_accuracy_meter.update(acc1[0].item(), ans.size(0))

print(f'Total Test Accuracy: {test_accuracy_meter.avg:.2f} ')
for cat, acc in cat_accuracy_meters.items():
    print(f'{cat} Test Accuracy: {acc.avg:.2f}')