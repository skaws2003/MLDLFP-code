import pandas as pd
import torch

DATASET_PATH = {
    'train': '../dataset/train.csv',
    'test': '../dataset/test.csv'
}

class Sentiment_dataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path):
        """
        Dataloader for Sentiment-140 dataset
        Fields
        - dataset_path: file path to dataset csv file
        """
        all_data = pd.read_csv(dataset_path,names=['polarity','id','date','query','user','text'])
        polarity = all_data['polarity']
        text = all_data['text']
        self.data = pd.DataFrame(data={'polarity':polarity,'text':text})

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self,index):
        rtn = self.data.iloc[index]
        return (rtn['text'],rtn['polarity'])


