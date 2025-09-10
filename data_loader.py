import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_data_loaders(dataset_path, image_size=(224, 224), batch_size=32, validation_split=0.1, test_split=0.1):
    """
    Prepares and returns training, validation, and test data loaders.
    
    Args:
        dataset_path (str): The path to the root of the dataset directory.
        image_size (tuple): The size to resize images to.
        batch_size (int): The number of images per batch.
        validation_split (float): The proportion of the dataset to use for validation.
        test_split (float): The proportion of the dataset to use for testing.
        
    Returns:
        tuple: A tuple containing the train_loader, val_loader, test_loader, and class names.
    """
    
    # Define transformations for training data with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for validation and test data (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)

    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    validation_size = int(validation_split * total_size)
    train_size = total_size - validation_size - test_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply appropriate transforms to the validation and test sets
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes

# Path to your organized dataset
dataset_path = '/Users/azratuncay/Desktop/hackathon/organized_dataset'