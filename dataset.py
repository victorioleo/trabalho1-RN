"""
Dataset utilities for CIFAR-100
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from cifar100_labels import get_coarse_label


class CIFAR100WithCoarse(Dataset):
    """
    CIFAR-100 dataset that also returns the coarse (superclass) label.
    """
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar100 = datasets.CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=download
        )
        
    def __len__(self):
        return len(self.cifar100)
    
    def __getitem__(self, idx):
        image, fine_label = self.cifar100[idx]
        coarse_label = get_coarse_label(fine_label)
        return image, fine_label, coarse_label


def get_cifar100_transforms(train=True):
    """Get standard data augmentation transforms for CIFAR-100."""
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])
    return transform


def get_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get train and test dataloaders for CIFAR-100.
    
    Returns:
        train_loader, test_loader
    """
    train_transform = get_cifar100_transforms(train=True)
    test_transform = get_cifar100_transforms(train=False)
    
    train_dataset = CIFAR100WithCoarse(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = CIFAR100WithCoarse(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
