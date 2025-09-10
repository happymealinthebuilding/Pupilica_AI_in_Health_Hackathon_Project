import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

# Import functions from other files
from model import build_model
from utils import preprocess_image, grad_cam_heatmap, visualize_results

# Define the list of class names based on your dataset
# Make sure this matches the order of your dataset folders
CLASS_NAMES = ['MEL', 'BCC', 'NV', 'AK', 'SK', 'VASC', 'DF'] 
NUM_CLASSES = len(CLASS_NAMES)

# Set the device dynamically for both CUDA (NVIDIA) and MPS (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU device")

@st.cache_resource
def load_model():
    """
    Loads the trained model with caching.
    
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = build_model(NUM_CLASSES)
    model_path = 'derm_ai_model.pth' # Make sure your trained model is in the same directory
    
    # Check if the model file exists
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please train the model and place it in the project root directory.")
        return None

# Load the model
model = load_model()

# --- Streamlit UI ---
st.set_page_config(
    page_title="Derm-AI: Skin Lesion Triage",
    page_icon="⚕️",
    layout="wide"
)

st.title("Derm-AI: An AI-Powered Skin Lesion Triage System ⚕️")
st.write("""
This application is designed to provide a preliminary, AI-driven assessment of skin lesions. 
It uses a deep learning model to classify a user-uploaded image and generates a heatmap 
to show which parts of the image the model focused on.
""")

st.warning("⚠️ **Medical Disclaimer:** This tool is for informational purposes only and is **not** a substitute for professional medical advice. Always consult a qualified healthcare professional for a definitive diagnosis.")

# Sidebar for an image of the model
# You can add an image file to your project and use it here.
# For example, create an image of a brain with a medical icon to add a visual flair.
st.sidebar.image("derm_ai_logo.png", use_column_width=True) # Replace 'derm_ai_logo.png' with your image file
st.sidebar.title("How it Works")
st.sidebar.info("""
1.  **Upload** a clear image of a skin lesion.
2.  Our model **predicts** the most likely class.
3.  We use **Grad-CAM** to generate a heatmap, highlighting the most important regions.
4.  The results are displayed, along with **confidence scores**, for a more transparent assessment.
""")

st.markdown("---")

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner('Analyzing image...'):
        # Preprocess the image for the model
        image_tensor = preprocess_image(image)
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(image_tensor.to(device))
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get predicted class and confidence
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        predicted_class_name = CLASS_NAMES[predicted_class_idx.item()]
        
        # Get confidence scores for all classes
        confidence_scores = {CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities)}

        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)

        with col2:
            st.write("### Predicted Diagnosis")
            st.success(f"**{predicted_class_name}**")
            st.write(f"Confidence: **{confidence.item():.2f}**")
            
        st.markdown("---")

        st.write("### Model's Focus: Grad-CAM Heatmap")
        
        # Generate Grad-CAM heatmap
        # For ResNet-50, a good target layer is the last convolutional block
        target_layer = model.layer4[-1]
        heatmap = grad_cam_heatmap(model, target_layer, image_tensor.to(device), use_cuda=torch.cuda.is_available())
        
        # Visualize and display results
        plot_buffer = visualize_results(image, heatmap, f"Prediction: {predicted_class_name}", confidence_scores, CLASS_NAMES)
        
        st.image(plot_buffer, caption='Heatmap and Confidence Scores', use_column_width=True)
        st.markdown("""
        The heatmap indicates the regions (in red/orange) that the model found most relevant
        when making its prediction. This can provide insight into the model's reasoning.
        """)