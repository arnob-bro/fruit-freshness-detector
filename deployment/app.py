"""
Streamlit Web Application for Fruit Freshness Detection
User-friendly interface for real-time predictions
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model


@st.cache_resource
def load_model(model_path, model_type='resnet'):
    """
    Load model with caching
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        
    Returns:
        Loaded model
    """
    if model_type == 'resnet':
        model = get_resnet_model(num_classes=1)
    elif model_type == 'mobilenet':
        model = get_mobilenet_model(num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def predict(model, image, model_type='resnet', threshold=0.5):
    """
    Make prediction on image
    
    Args:
        model: PyTorch model
        image: PIL Image
        model_type: Type of model
        threshold: Classification threshold
        
    Returns:
        dict: Prediction results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        proba = torch.sigmoid(output).item() if output.dim() == 1 else output.item()
    
    prediction = 'Fresh' if proba > threshold else 'Rotten'
    confidence = proba if prediction == 'Fresh' else 1 - proba
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probability': proba
    }


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Fruit Freshness Detector",
        page_icon="🍎",
        layout="wide"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    st.title("🍎 Fruit Freshness Detection System")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["resnet", "mobilenet"],
        index=0
    )
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/saved/best_model.pth"
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    show_gradcam = st.sidebar.checkbox("Show GradCAM Explanation", value=False)
    
    # Load model
    try:
        model = load_model(model_path, model_type)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fruit image...",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("Prediction Results")
        
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                result = predict(model, image, model_type, threshold)
            
            # Display results
            st.subheader("Prediction")
            if result['prediction'] == 'Fresh':
                st.success(f"✅ {result['prediction']}")
            else:
                st.error(f"❌ {result['prediction']}")
            
            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            
            # Progress bar
            st.progress(result['confidence'])
            
            # Probability distribution
            st.subheader("Probability Distribution")
            prob_data = {
                'Rotten': 1 - result['probability'],
                'Fresh': result['probability']
            }
            st.bar_chart(prob_data)
            
            # GradCAM explanation
            if show_gradcam:
                st.subheader("Model Explanation (GradCAM)")
                try:
                    from explainability.gradcam import GradCAM
                    import torch.nn.functional as F
                    
                    # Get target layer
                    if model_type == 'resnet':
                        target_layer = model.backbone.layer4[-1]
                    elif model_type == 'mobilenet':
                        target_layer = model.backbone.features[-1]
                    
                    # Generate CAM
                    gradcam = GradCAM(model, target_layer)
                    input_tensor = preprocess_image(image).to(device)
                    cam = gradcam.generate_cam(input_tensor)
                    overlayed = gradcam.overlay_cam(image, cam)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.image(image, caption="Original", use_container_width=True)
                    with col_b:
                        st.image(cam, caption="Attention Map", use_container_width=True, clamp=True)
                    with col_c:
                        st.image(overlayed, caption="Overlay", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate GradCAM: {str(e)}")
        else:
            st.info("Please upload an image to get predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses deep learning to detect fruit freshness.
    Upload an image of a fruit to get real-time predictions.
    
    **Models Available:**
    - ResNet50 (Transfer Learning)
    - MobileNetV2 (Lightweight)
    """)


if __name__ == '__main__':
    main()

