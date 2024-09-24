import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
import torch.nn.functional as F
import numpy as np
import re
from tqdm import tqdm
import clip

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def preprocessing(text):
  input_text = text
  input_text = input_text.lower()

  # Removing periods except if it occurs as decimal
  input_text = re.sub(r'(?<!\d)\.(?!\d)', '', input_text)

  # Converting number words to digits
  number_words = {
      "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
      "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
      "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
      "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
      "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
      "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
      "eighty": "80", "ninety": "90"
  }
  input_text = re.sub(r'\b(?:' + '|'.join(number_words.keys()) + r')\b', lambda x: number_words[x.group()], input_text)

  # Removing articles (a, an, the)
  if len(input_text)>3:
    input_text = re.sub(r'\b(?:a|an|the)\b', '', input_text)

  # Adding apostrophe if a contraction is missing it
  input_text = re.sub(r'\b(\w+(?<!e)(?<!a))nt\b', r"\1n't", input_text)

  # input_text = re.sub(r'\b(\w+(?<!t))ent\b', r"\1en't", input_text)

  # Replacing all punctuation (except apostrophe and colon) withinput_text a space character
  input_text = re.sub(r'[^\w\':]|(?<=\d),(?=\d)', ' ', input_text)

  # Removing extra spaces
  input_text = re.sub(r'\s+', ' ', input_text).strip()

  return input_text


class TrainDataset(Dataset):
    IMAGE_PATH = {
        "train": { 
             "questions": "v2_OpenEnded_mscoco_train2014_questions.json",
             "answers":  "v2_mscoco_train2014_annotations.json",
             "img_folder": "train2014"
             },
        "val": {
            "questions": "v2_OpenEnded_mscoco_val2014_questions.json", 
            "answers": "v2_mscoco_val2014_annotations.json",
            "img_folder": "val2014"
        }
    }
    def __init__(self, root, subset, transform=None):
        self.subset = subset
        self.root = root
        self.transform = transform
        self.selection = most_common_from_dict
        q_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]["questions"]))

        with open(q_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(
                lambda x: f"{self.IMAGE_PATH[subset]["img_folder"]}/COCO_{self.IMAGE_PATH[subset]["img_folder"]}_{x:012d}.jpg")
        path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[subset]['answers']))
        with open(path, 'r') as f:
                    data = json.load(f)
        df_annotations = pd.DataFrame(data["annotations"])
        self.vocab={}
        i=0
        with open('common_vocab.txt', 'r') as file:
                for line in file:
                    self.vocab[line[:-1]]=i
                    i+=1
        #    print(self.vocab)
        indices=[]
        for i in range(len(df_annotations)):
                selected_answer = self.selection(df_annotations["answers"][i])
                selected_answer = preprocessing(selected_answer)
                if selected_answer not in self.vocab.keys():
                    indices.append(i)
                    # print(selected_answer)
        df_annotations.drop(indices,axis=0,inplace=True)
        df_annotations.reset_index(inplace=True,drop=True) 
        #    print(df_annotations)
        df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='right')
        df["image_id"] = df["image_id_x"]
        if not all(df["image_id_y"] == df["image_id_x"]):
                    print("There is something wrong with image_id")
        del df["image_id_x"]
        del df["image_id_y"]
        self.df = df
        self.n_samples = self.df.shape[0]

        
    def __getitem__(self, index):
        image_path = self.df["image_path"][index]
        question = self.df["question"][index]
        selected_answers = preprocessing(self.selection(self.df["answers"][index]))
       
        image_path = os.path.expanduser(os.path.join(self.root, image_path))
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        answer = torch.tensor(self.vocab[selected_answers])
        return {"img": img, "question": question, "answer": answer, "domain": "source"}
    
    def __len__(self):
        return len(self.df["answers"])