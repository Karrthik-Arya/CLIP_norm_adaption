import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torch. multiprocessing as mp
import clip
from transformers import OFATokenizer, OFAModel
from torchvision import transforms

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# mp.set_start_method('spawn')
model, preprocess = clip.load("ViT-B/32", device=device)

# ckpt_dir = "../ofa-large"
# ofa_tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
# ofa_model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
# ofa_model.to(device)
# ofa_model.eval()

# mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
# resolution = 256
# patch_resize_transform = transforms.Compose([
#     lambda image: image.convert("RGB"),
#     transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=mean, std=std)
# ])

# def generate_caption(image):
#     """Generate caption for an image using the OFA model."""
#     with torch.no_grad():
#         inputs = ofa_tokenizer("what does the image describe?", return_tensors="pt").to(device)
#         images = patch_resize_transform(image).unsqueeze(0).to(device)
#         outputs = ofa_model.generate(**inputs, patch_images=images)
#         caption = ofa_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#     return caption

class TestDataset(Dataset):
    def __init__(self, img_path, questions_path, captions_path):
        df = pd.read_csv(questions_path)
        self.captions = pd.read_csv(captions_path)
        self.captions.set_index("id", inplace=True)

        self.img_path = img_path
        self.vocab={}
        i=0
        with open('common_vocab.txt', 'r') as file:
            for line in file:
                self.vocab[line[:-1]]=i
                i+=1
        indices=[]
        for i in range(len(df)):
                selected_answer = df["answer"][i]
                if selected_answer not in self.vocab.keys():
                    indices.append(i)
        df.drop(indices,axis=0,inplace=True)
        df.reset_index(inplace=True,drop=True) 
        self.df = df
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.df["image"][index]
        question = self.df["question"][index]
        selected_answer = self.df["answer"][index]
        caption = self.captions.loc[image_path, "caption"]
       
        image_path = os.path.expanduser(os.path.join(self.img_path, image_path))
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        answer = torch.tensor(self.vocab[selected_answer])
        return {"img": img, "question": question, "answer": answer, "img_path" : self.df["image"][index], "caption": caption}