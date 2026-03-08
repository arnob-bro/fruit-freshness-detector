"""
GradCAM Implementation for Model Explainability
Visualizes which parts of the image the model focuses on
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM)
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM
        
        Args:
            model: PyTorch model
            target_layer: Target layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate CAM for given input
        
        Args:
            input_image: Input image tensor
            class_idx: Class index (None for predicted class)
            
        Returns:
            CAM heatmap
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Compute CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def overlay_cam(self, image, cam, alpha=0.4):
        """
        Overlay CAM on original image
        
        Args:
            image: Original image (PIL Image or numpy array)
            cam: CAM heatmap
            alpha: Transparency factor
            
        Returns:
            Overlaid image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam_resized = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return overlayed


def visualize_gradcam(model, image_path, model_type='resnet', save_path=None):
    """
    Visualize GradCAM for a given image
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        model_type: Type of model ('resnet' or 'mobilenet')
        save_path: Path to save visualization
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Get target layer
    if model_type == 'resnet':
        target_layer = model.backbone.layer4[-1]
    elif model_type == 'mobilenet':
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    overlayed = gradcam.overlay_cam(image, cam)
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GradCAM visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def explain_prediction(model, image_path, model_type='resnet', threshold=0.5):
    """
    Explain model prediction for an image
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        model_type: Type of model
        threshold: Classification threshold
        
    Returns:
        dict: Prediction and explanation
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        proba = torch.sigmoid(output).item() if output.dim() == 1 else output.item()
        prediction = 'Fresh' if proba > threshold else 'Rotten'
    
    # Get target layer
    if model_type == 'resnet':
        target_layer = model.backbone.layer4[-1]
    elif model_type == 'mobilenet':
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    return {
        'prediction': prediction,
        'confidence': proba if prediction == 'Fresh' else 1 - proba,
        'cam': cam,
        'image': image
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GradCAM visualization')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_type', type=str, default='resnet',
                        choices=['resnet', 'mobilenet'],
                        help='Type of model')
    parser.add_argument('--save_path', type=str, default='gradcam_visualization.png',
                        help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Load model
    if args.model_type == 'resnet':
        model = get_resnet_model(num_classes=1)
    else:
        model = get_mobilenet_model(num_classes=1)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize
    visualize_gradcam(model, args.image, args.model_type, args.save_path)



