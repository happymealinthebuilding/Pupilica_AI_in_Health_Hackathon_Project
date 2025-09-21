#model.py
import torch
import torch.nn as nn
from torchvision import models

def fine_tune_model(num_classes):
    """
    Loads a pre-trained EfficientNet-V2-M model and replaces its classifier head.
    
    Args:
        num_classes (int): The number of output classes for the new classifier.
        
    Returns:
        torch.nn.Module: The fine-tuned model ready for training.
    """
    model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classifier layer with a new one
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model