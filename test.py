import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from datasets.testDatset import TestDataset
from tqdm import tqdm
import copy

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
    def __init__(self,num_classes=1000):
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

        self.classifier = nn.Linear(1024,num_classes)


    def forward(self, image, text):

        if "source" in text:
            inputs1 = clip.tokenize(text["source"]).to(self.device)
            image_features1 = self.source_model.encode_image(image["source"].to(self.device))
            text_features1 = self.source_model.encode_text(inputs1)

        if "target" in text:
            inputs2 = clip.tokenize(text["target"]).to(self.device)
            image_features2 = self.target_model.encode_image(image["target"].to(self.device))
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
transfer_model = transfer_model.to('cuda:1')
transfer_model.load_state_dict(torch.load('./clip_vqa_v2.pth'))
transfer_model.eval()   

test_dataset = TestDataset('data/test/images', 'data/test/test_questions.csv')

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

    output = transfer_model(img,ques)
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
        pred = torch.stack(cat_output[cat]).to("cuda:1")
        if(answer.size(0) > 0):
            acc1 = accuracy(pred, answer, topk=(1,))
            cat_accuracy_meters[cat].update(acc1[0].item(), answer.size(0))

    acc1 = accuracy(output, ans, topk=(1,))
    test_accuracy_meter.update(acc1[0].item(), ans.size(0))

print(f'Total Test Accuracy: {test_accuracy_meter.avg:.2f} ')
for cat, acc in cat_accuracy_meters.items():
    print(f'{cat} Test Accuracy: {acc.avg:.2f}')