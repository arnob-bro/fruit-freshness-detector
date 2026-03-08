"""
Training Script for Fruit Freshness Detection
Supports training CNN from scratch, ResNet, and MobileNet models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import get_loaders, get_class_names
from utils.metrics import evaluate_model
from utils.visualization import plot_training_history
from models.cnn_scratch import get_model as get_cnn_model
from models.resnet_transfer import get_model as get_resnet_model
from models.mobilenet_transfer import get_model as get_mobilenet_model


class Trainer:
    """Training class for fruit freshness detection models"""
    
    def __init__(self, config_path='training/config.yaml'):
        """
        Initialize trainer
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = get_loaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )
        
        self.class_names = get_class_names(self.config['data']['data_dir'])
        print(f"Class names: {self.class_names}")
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
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
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        optimizer_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config['training'].get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=self.config['training'].get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config['training'].get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            preds = (outputs > 0.5).long()
            total += labels.size(0)
            correct += (preds == labels.long()).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        metrics, _, _, _ = evaluate_model(self.model, self.val_loader, self.device)
        return metrics['f1_score'], metrics['accuracy']
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        best_val_acc = 0.0
        save_dir = self.config['training']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("="*50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-"*50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics, _, _, _ = evaluate_model(self.model, self.val_loader, self.device)
            val_loss = 1 - val_metrics['f1_score']  # Approximate validation loss
            val_acc = val_metrics['accuracy'] * 100
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print statistics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, model_path)
                print(f"Saved best model with Val Acc: {val_acc:.2f}%")
        
        # Save final model and history
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, final_model_path)
        
        # Plot training history
        plot_training_history(self.history, save_path=os.path.join(save_dir, 'training_history.png'))
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fruit freshness detection model')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    trainer = Trainer(config_path=args.config)
    trainer.train()


if __name__ == '__main__':
    main()



