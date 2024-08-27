import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

# test_df = 
# answer_counts = test_df['answer'].value_counts()
# weights = [1/answer_counts[i] for i in test_df['answer'].values]

class TestDataset(Dataset):
    def __init__(self, img_path, questions_path):
        df = pd.read_csv(questions_path)
        self.question = df['question']
        self.image_id = df['image']
        self.img_path = img_path
        self.vocab={}
        i=0
        with open('common_vocab.txt', 'r') as file:
            for line in file:
                self.vocab[line[:-1]]=i
                i+=1
        indices=[]
        for i in range(len(df)):
                selected_answer = df["answers"][i]
                if selected_answer not in self.vocab.keys():
                    indices.append(i)
        df.drop(indices,axis=0,inplace=True)
        df.reset_index(inplace=True,drop=True) 
        

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, index):
        ques = self.question.iloc[index]
        answer = self.answer.iloc[index]
        image_id = self.image_id.iloc[index]
        img = Image.open(f'f{self.img_path}/{image_id}').convert('RGB')
        return {"img": img, "question": ques, "answer": answer, "domain": "target"}