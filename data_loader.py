#data_loader.py
import os
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

def get_data_loaders(data_dir, batch_size=32):
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Path to the root directory containing the train/val/test folders.
        batch_size (int): The batch size for the DataLoaders.
    
    Returns:
        tuple: A tuple containing the dataloaders dictionary and a list of class names.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    
    # Weighted Random Sampler for Training Data
    train_targets = np.array(image_datasets['train'].targets)
    class_counts = collections.Counter(train_targets)
    num_samples = sum(class_counts.values())
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[t] for t in train_targets])
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    class_names = image_datasets['train'].classes
    return dataloaders, class_names