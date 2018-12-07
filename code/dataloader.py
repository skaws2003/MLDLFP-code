import our_dataset as od
import torch
import torchvision

datasets = {
    'test': od.Polarity_dataset(pos_path=od.DATASET_PATH['positive_test'],neg_path=od.DATASET_PATH['negative_test']),
    'train': od.Polarity_dataset(pos_path=od.DATASET_PATH['positive_train'],neg_path=od.DATASET_PATH['negative_train'])
}

dataloaders = {
    'test': torch.utils.data.DataLoader(
        dataset=datasets['test'],
        batch_size=1,
        shuffle=False
    ),
    'train': torch.utils.data.DataLoader(
        dataset=datasets['train'],
        batch_size=1,
        shuffle=True
    )
}

if __name__ == "__main__":
    for i,(text,semantic) in enumerate(dataloaders['train']):
        if i > 2:
            break
        print(text)
        print(semantic)