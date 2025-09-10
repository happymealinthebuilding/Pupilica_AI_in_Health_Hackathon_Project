import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Set the device dynamically for both CUDA (NVIDIA) and MPS (Apple Silicon)
# Using Apple Silicon due to no GPU available in the current environment. Will switch to cuda once a compatible GPU is available on Google Colab.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU device")

def build_model(num_classes, fine_tune=True):
    """
    Builds a pre-trained ResNet-50 model and fine-tunes it.
    
    Args:
        num_classes (int): The number of output classes.
        fine_tune (bool): If True, fine-tunes the entire model. If False, only the final layer.
        
    Returns:
        torch.nn.Module: The configured PyTorch model.
    """
    model = models.resnet50(pretrained=True)

    # Freeze all layers if not fine-tuning
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Trains the provided model.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to train the model on (e.g., 'cuda' or 'cpu').
        
    Returns:
        tuple: A tuple containing the trained model and training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history