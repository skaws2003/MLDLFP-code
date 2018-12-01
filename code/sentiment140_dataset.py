import torch
import torchvision
import pandas as pd

# Some args
DATASET_PATH = {
    'train': 'dataset/train.csv',
    'test': 'dataset/test.csv'
}

class Sentiment_dataset(torch.utils.data.Dataset):
    def __init__(self,filename,print_line=None):
        if print_line:
            print(print_line)
        all_data = pd.read_csv(filename,names=['polarity','id','date','query','user','text'])
        polarity = all_data['polarity']
        text = all_data['text']
        self.data = pd.DataFrame(data={'polarity':polarity,'text':text})

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self,index):
        rtn = self.data.iloc[index]
        return (rtn['text'],rtn['polarity'])