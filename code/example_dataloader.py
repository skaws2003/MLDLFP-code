import sentiment140_dataset as sd
import torch

datasets = {x: sd.Sentiment_dataset(filename=sd.DATASET_PATH[x]) for x in ['test','train']}

dataloaders = {
    'test': torch.utils.data.DataLoader(
        dataset=datasets['test'],
        batch_size=1,
        shuffle=False
    ),
    'train': torch.utils.data.DataLoader(
        dataset=datasets['train'],
        batch_size=10,
        shuffle=True
    )
}


for i,(text,semantic) in enumerate(dataloaders['train']):
    if i > 2:
        break
    print(text)
    print(semantic)