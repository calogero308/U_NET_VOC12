from torch.utils.data import DataLoader
import torchvision.transforms as Ts
from torchvision.transforms import InterpolationMode
from .dataset import VOCSegmentationDataset
from torch.utils.data import random_split, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

#Trasformations |Training|
train_transform = A.Compose([
    A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

#|Validation|
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

#Loading the full training dataset, without transformation
full_dataset_training = VOCSegmentationDataset(
    root_dir="data/archive/VOC2012_train_val/VOC2012_train_val",
    transform=None,
    has_masks=True
)

#Define the train dataset with the augmentation transformation
train_dataset = VOCSegmentationDataset(root_dir="data/archive/VOC2012_train_val/VOC2012_train_val",
                            transform=train_transform,
                            has_masks=True)

#Define the validation dataset with only the resize and normalization transformation
val_dataset = VOCSegmentationDataset(root_dir="data/archive/VOC2012_train_val/VOC2012_train_val",
                                    transform=val_transform,
                                    has_masks=True)

#Define the test dataset
test_dataset = VOCSegmentationDataset(root_dir="data/archive/VOC2012_test/VOC2012_test",
                            transform=Ts.Compose([
                                Ts.Resize((256,256), interpolation=InterpolationMode.BILINEAR),
                                Ts.ToTensor()
                            ]),
                            has_masks=False)

#Function that create and return the train dataloader
def get_train_loader(train_dataset=train_dataset, BATCH_SIZE=4):

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_loader

#Function that create and return the test dataloader
def get_test_loader(train_dataset=test_dataset, BATCH_SIZE=4):

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_loader



#DATASET
#Splitting the train dataset as 80% train images and 20% validation images
train_size = int(0.8 * len(full_dataset_training))
val_size = len(full_dataset_training) - train_size

#Reproducibility
generator = torch.Generator().manual_seed(42)

#Indices used for dataset creation
train_indices, val_indices = random_split(
    range(len(full_dataset_training)),
    [train_size, val_size],
    generator=generator  # opzionale
)

train_dataset = Subset(train_dataset, train_indices.indices)    #Train dataset -> 80% of original dataset
val_dataset   = Subset(val_dataset, val_indices.indices)        #Validation dataset -> 20% of original dataset

#Function that creates and returns the train and validation dataloader 
def get_Train_Val_loader_split(data_set=train_dataset, BATCH_SIZE=4):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader
