#train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import os
from google.colab import drive

from data_loader import get_data_loaders
from model import fine_tune_model
from utils import plot_confusion_matrix

def train_model(model, dataloaders, num_epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("Starting model training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=f'Epoch {epoch+1} {phase}'):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Training complete!")
    return model

def evaluate_model(model, dataloader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluation'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    # You will need to mount Google Drive or ensure these paths exist
    drive.mount('/content/drive')
    
    split_data_path = '/content/split_dataset'
    
    # Placeholder for the class names, as `data_preparation.py` would have been run already
    ground_truth_path = '/content/ISIC_2019_Training_GroundTruth.csv'
    df = pd.read_csv(ground_truth_path)
    class_names = [col for col in df.columns if col not in ['image', 'UNK']]
    
    dataloaders, class_names = get_data_loaders(split_data_path)
    model = fine_tune_model(len(class_names))
    model = train_model(model, dataloaders, num_epochs=50)
    
    model_save_path = '/content/derm_ai_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    evaluate_model(model, dataloaders['test'], class_names)