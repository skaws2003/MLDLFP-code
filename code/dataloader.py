import our_dataset as od
import torch
import torchvision

datasets = {
    'test': od.Polarity_dataset(
        pos_path=od.DATASET_PATH['positive_test'],
        neg_path=od.DATASET_PATH['negative_test'],
        one_hot=False),
    'train': od.Polarity_dataset(
        pos_path=od.DATASET_PATH['positive_train'],
        neg_path=od.DATASET_PATH['negative_train'],
        one_hot=False)
}

dataloaders = {
    'test': od.Polarity_dataloader(datasets['test'],batch_size=2),
    'train': od.Polarity_dataloader(datasets['train'],batch_size=2)
}

print(len(dataloaders['train']))

if __name__ == "__main__":
    for i,data in enumerate(dataloaders['train']):
        if i > 0:
            break
        print(data[1].type())