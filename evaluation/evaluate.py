"""
Evaluation Script for Fruit Freshness Detection Models
Comprehensive evaluation with multiple metrics
"""

import torch
import torch.nn as nn
import yaml
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import get_loaders, get_class_names
from utils.metrics import evaluate_model, print_classification_report, plot_roc_curve
from utils.visualization import plot_confusion_matrix, visualize_predictions
from models.cnn_scratch import get_model as get_cnn_model
from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model


class Evaluator:
    """Evaluation class for fruit freshness detection models"""
    
    def __init__(self, model_path, config_path='training/config.yaml'):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to saved model checkpoint
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        _, _, self.test_loader = get_loaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )
        
        self.class_names = get_class_names(self.config['data']['data_dir'])
        
        # Load model
        self.model = self._create_model()
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _create_model(self):
        """Create model based on config"""
        model_type = self.config['model']['type']
        
        if model_type == 'cnn':
            return get_cnn_model(num_classes=1)
        elif model_type == 'resnet':
            return get_resnet_model(
                num_classes=1,
                pretrained=self.config['model']['pretrained'],
                freeze_backbone=self.config['model']['freeze_backbone']
            )
        elif model_type == 'mobilenet':
            return get_mobilenet_model(
                num_classes=1,
                pretrained=self.config['model']['pretrained'],
                freeze_backbone=self.config['model']['freeze_backbone']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    def evaluate(self, threshold=0.5, save_dir='evaluation/results'):
        """
        Evaluate model on test set
        
        Args:
            threshold (float): Classification threshold
            save_dir (str): Directory to save evaluation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print("Evaluating Model on Test Set")
        print("="*50)
        
        # Evaluate
        metrics, y_true, y_pred, y_proba = evaluate_model(
            self.model, 
            self.test_loader, 
            self.device, 
            threshold=threshold
        )
        
        # Print metrics
        print("\nTest Set Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Print classification report
        print_classification_report(y_true, y_pred, self.class_names)
        
        # Plot confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, self.class_names, save_path=cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Plot ROC curve
        if 'roc_auc' in metrics:
            roc_path = os.path.join(save_dir, 'roc_curve.png')
            plot_roc_curve(y_true, y_proba, save_path=roc_path)
            print(f"ROC curve saved to {roc_path}")
        
        # Visualize predictions
        pred_path = os.path.join(save_dir, 'sample_predictions.png')
        visualize_predictions(
            self.model, 
            self.test_loader, 
            self.device, 
            self.class_names,
            num_samples=8,
            save_path=pred_path
        )
        print(f"Sample predictions saved to {pred_path}")
        
        # Save metrics to file
        import json
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate fruit freshness detection model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                        help='Path to config file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--save_dir', type=str, default='evaluation/results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(args.model, args.config)
    evaluator.evaluate(threshold=args.threshold, save_dir=args.save_dir)


if __name__ == '__main__':
    main()



