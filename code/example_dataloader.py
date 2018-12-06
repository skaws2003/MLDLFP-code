import code.our_dataset as od
import torch
import torchvision

datasets = {x: od.Sentiment_dataset(dataset_path=od.DATASET_PATH[x]) for x in ['train','test']}

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