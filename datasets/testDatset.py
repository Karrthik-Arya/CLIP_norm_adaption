import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import clip

# test_df = 
# answer_counts = test_df['answer'].value_counts()
# weights = [1/answer_counts[i] for i in test_df['answer'].values]

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class TestDataset(Dataset):
    def __init__(self, img_path, questions_path):
        df = pd.read_csv(questions_path)
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
       
        image_path = os.path.expanduser(os.path.join(self.img_path, image_path))
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        answer = torch.tensor(self.vocab[selected_answer])
        return {"img": img, "question": question, "answer": answer}