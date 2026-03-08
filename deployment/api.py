"""
REST API for Fruit Freshness Detection
FastAPI-based API for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model

app = FastAPI(
    title="Fruit Freshness Detection API",
    description="API for detecting fruit freshness using deep learning",
    version="1.0.0"
)

# Global model variable
model = None
model_type = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: str, model_t: str = 'resnet'):
    """
    Load model
    
    Args:
        model_path: Path to model checkpoint
        model_t: Type of model
    """
    global model, model_type
    
    if model_t == 'resnet':
        model = get_resnet_model(num_classes=1)
    elif model_t == 'mobilenet':
        model = get_mobilenet_model(num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_t}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    model_type = model_t
    
    return model


def preprocess_image(image_bytes: bytes):
    """
    Preprocess image for model input
    
    Args:
        image_bytes: Image bytes
        
    Returns:
        Preprocessed tensor
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def predict(image_bytes: bytes, threshold: float = 0.5):
    """
    Make prediction on image
    
    Args:
        image_bytes: Image bytes
        threshold: Classification threshold
        
    Returns:
        dict: Prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_tensor = preprocess_image(image_bytes).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            proba = torch.sigmoid(output).item() if output.dim() == 1 else output.item()
        
        prediction = 'Fresh' if proba > threshold else 'Rotten'
        confidence = proba if prediction == 'Fresh' else 1 - proba
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'probability': round(proba, 4),
            'threshold': threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    # Default model path - can be changed via environment variable
    import os
    model_path = os.getenv('MODEL_PATH', 'models/saved/best_model.pth')
    model_t = os.getenv('MODEL_TYPE', 'resnet')
    
    try:
        load_model(model_path, model_t)
        print(f"Model loaded successfully: {model_t}")
    except Exception as e:
        print(f"Warning: Could not load model: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fruit Freshness Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    threshold: float = 0.5
):
    """
    Predict fruit freshness from uploaded image
    
    Args:
        file: Uploaded image file
        threshold: Classification threshold (0.0-1.0)
        
    Returns:
        JSON response with prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please load model first.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = predict(image_bytes, threshold)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/load_model")
async def load_model_endpoint(
    model_path: str,
    model_type: str = 'resnet'
):
    """
    Load model from specified path
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('resnet' or 'mobilenet')
        
    Returns:
        JSON response with status
    """
    try:
        load_model(model_path, model_type)
        return {
            "status": "success",
            "message": f"Model loaded successfully: {model_type}",
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/model_info")
async def model_info():
    """Get information about loaded model"""
    if model is None:
        return {
            "model_loaded": False,
            "message": "No model loaded"
        }
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_loaded": True,
        "model_type": model_type,
        "device": str(device),
        "total_parameters": num_params,
        "trainable_parameters": trainable_params
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



