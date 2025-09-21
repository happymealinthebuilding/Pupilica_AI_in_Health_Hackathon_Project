#app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import io
import os
import pandas as pd
from pyngrok import ngrok
import time
from torchvision.transforms import transforms

from model import fine_tune_model
from utils import GradCAMpp, ScoreCAM, overlay_heatmap

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image, transform(image).unsqueeze(0)

@st.cache_resource
def load_model_and_classes():
    try:
        model = fine_tune_model(num_classes=8)  # Assuming 8 classes from your CSV
        ground_truth_path = '/content/ISIC_2019_Training_GroundTruth.csv'
        df = pd.read_csv(ground_truth_path)
        class_names = [col for col in df.columns if col not in ['image', 'UNK']]
        
        # Re-create the model with the correct number of classes
        model = fine_tune_model(num_classes=len(class_names))
        
        model_path = '/content/derm_ai_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, class_names
    except Exception as e:
        st.error(f'Error loading model or class data: {e}')
        return None, None

def main():
    model, class_names = load_model_and_classes()
    if model is None:
        st.stop()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    grad_cam_pp = GradCAMpp(model)
    score_cam = ScoreCAM(model)

    st.set_page_config(page_title='Derm-AI: Skin Lesion Triage', layout='wide')

    # --- Logo and Title Layout ---
    col1, col2 = st.columns([1, 8])
    with col1:
        if os.path.exists('/content/derm_ai_logo.png'):
            st.image('/content/derm_ai_logo.png', width=100)
    with col2:
        st.title('Derm-AI: Skin Lesion Triage System')
        st.markdown('An AI-powered tool for preliminary skin lesion assessment.')
    st.markdown('---')
    
    with st.expander('⚠️ **Important Disclaimer**'):
        st.warning('**Derm-AI is for educational and informational purposes only.** This application is an AI-powered tool and is **NOT** a substitute for a professional medical diagnosis. **Always consult a qualified dermatologist or healthcare professional.**')

    st.subheader('Upload a Skin Lesion Image')
    uploaded_file = st.file_uploader('Drag and drop or click to upload an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        with st.spinner('Analyzing image...'):
            original_image_bytes = uploaded_file.getvalue()
            pil_image, input_tensor = preprocess_image(original_image_bytes)
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_class_idx = torch.topk(probabilities, 1)
            predicted_class_name = class_names[top_class_idx.item()]
            confidence_score = top_prob.item()

            grad_cam_pp_heatmap = grad_cam_pp(input_tensor, target_class=top_class_idx.item())
            score_cam_heatmap = score_cam(input_tensor, target_class=top_class_idx.item())

            grad_cam_pp_image = overlay_heatmap(pil_image, grad_cam_pp_heatmap)
            score_cam_image = overlay_heatmap(pil_image, score_cam_heatmap)

        st.success('Analysis Complete!')
        st.subheader('Analysis Results')
        st.markdown(f'**Primary AI Assessment:** `{predicted_class_name}`')
        st.markdown(f'**Confidence Score:** `{confidence_score:.2%}`')
        
        col1, col2, col3 = st.columns(3)
        with col1: st.image(pil_image, caption='Original Image', use_container_width=True)
        with col2: st.image(grad_cam_pp_image, caption='AI Focus: Grad-CAM++ Heatmap', use_container_width=True)
        with col3: st.image(score_cam_image, caption='AI Focus: Score-CAM Heatmap', use_container_width=True)
        
        st.markdown('''
            **What these heatmaps mean:** The heatmaps are visual explanations of the AI's decision-making. The red and yellow areas indicate the regions the model focused on to make its prediction. **Grad-CAM++** often provides sharper, more localized heatmaps, while **Score-CAM** uses a different approach to provide a more holistic view of the features contributing to the prediction. Comparing them side-by-side offers a more robust way to interpret the AI's reasoning.
        ''')
        st.info('**Final Reminder:** Please consult with a healthcare professional or dermatologist.')

    st.markdown('---')
    st.subheader('Ethical Chatbot (Prototype)')
    user_query = st.text_input('Ask me about general information related to skin lesions.')
    if user_query:
        with st.spinner('Thinking...'):
            st.info('I am an ethical AI assistant. I can provide general information, but **I cannot provide medical advice.** Always consult a doctor for an accurate diagnosis.')

if __name__ == '__main__':
    # This is a placeholder for the Pyngrok part in the original script
    # It won't work locally but demonstrates the structure for a Colab environment.
    ngrok.kill()
    ngrok.set_auth_token("2rca4ybNvjlXlIUzgHPxDWBK6tl_XWXjDK36dz5kcHvr3AqP")
    !streamlit run app.py &>/dev/null&
    url = ngrok.connect(addr="8501", proto="http")
    print(f"Streamlit app is available at: {url}")