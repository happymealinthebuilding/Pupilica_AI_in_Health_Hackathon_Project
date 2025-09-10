import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import io
from torchvision import transforms

def grad_cam_heatmap(model, target_layer, input_tensor, use_cuda=False):
    """
    Generates a Grad-CAM heatmap for a given input tensor.
    
    Args:
        model (torch.nn.Module): The trained model.
        target_layer (torch.nn.Module): The target convolutional layer.
        input_tensor (torch.Tensor): The input image tensor.
        use_cuda (bool): Whether to use CUDA for computation.
        
    Returns:
        np.ndarray: The generated heatmap.
    """
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=input_tensor)
    
    # Resize the heatmap to match the input image size
    heatmap = np.array(Image.fromarray(grayscale_cam[0, :]).resize(input_tensor.shape[2:][::-1]))
    return heatmap

def visualize_results(image, heatmap, title, confidence_scores, class_names):
    """
    Visualizes the original image with the Grad-CAM heatmap and confidence scores.
    
    Args:
        image (PIL.Image): The original input image.
        heatmap (np.ndarray): The Grad-CAM heatmap.
        title (str): The title for the plot.
        confidence_scores (dict): A dictionary of class names and their confidence scores.
        class_names (list): A list of the class names.
        
    Returns:
        io.BytesIO: A buffer containing the saved plot image.
    """
    img_np = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, heatmap, use_rgb=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(cam_image)
    ax1.set_title(title)
    ax1.axis('off')

    labels = list(confidence_scores.keys())
    scores = list(confidence_scores.values())

    ax2.barh(labels, scores, color='skyblue')
    ax2.set_xlabel("Confidence Score")
    ax2.set_title("Model Confidence Scores")
    ax2.set_xlim(0, 1)

    for index, value in enumerate(scores):
        ax2.text(value, index, f'{value:.2f}', va='center')

    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf

def preprocess_image(image_pil, image_size=(224, 224)):
    """
    Prepares a PIL image for model inference.
    
    Args:
        image_pil (PIL.Image): The input image.
        image_size (tuple): The size to resize the image to.
        
    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image_pil).unsqueeze(0)