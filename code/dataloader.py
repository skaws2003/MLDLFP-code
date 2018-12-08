import our_dataset as od
import torch
import torchvision

datasets = {
    'test': od.Polarity_dataset(pos_path=od.DATASET_PATH['positive_test'],neg_path=od.DATASET_PATH['negative_test']),
    'train': od.Polarity_dataset(pos_path=od.DATASET_PATH['positive_train'],neg_path=od.DATASET_PATH['negative_train'])
}

dataloaders = {
    'test': od.Polarity_dataloader(datasets['test']),
    'train': od.Polarity_dataloader(datasets['train'])
}

if __name__ == "__main__":
    for i,data in enumerate(dataloaders['train']):
        if i > 2:
            break
        print(data[0][0])
        print(data[0][1])