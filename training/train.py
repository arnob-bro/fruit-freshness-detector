"""
Training Script for Fruit Freshness Detection
Trains 5 CNN architectures per fruit (15 models total)
Uses transfer learning with 2-phase training: frozen backbone then fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import os
import sys
import json
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import get_loaders, get_class_names, get_class_weights
from utils.metrics import evaluate_model
from utils.visualization import plot_training_history
from models import get_model_by_name, ARCHITECTURES


class Trainer:
    """Training class for per-fruit, multi-architecture freshness models"""

    def __init__(self, config_path='training/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.fruits = self.config['fruits']
        self.arch_names = self.config['architectures']
        self.save_dir = self.config['training']['save_dir']

    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).long()
            total += labels.size(0)
            correct += (preds == labels.long()).sum().item()

        return running_loss / len(train_loader), 100 * correct / total

    def _train_model(self, fruit, arch_name):
        """Train a single model (both phases)."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  Training {ARCHITECTURES[arch_name][1]} for {fruit.upper()}")
        print(f"{'='*60}")

        # Data
        train_loader, val_loader, _ = get_loaders(
            data_dir=cfg['data']['data_dir'],
            batch_size=cfg['training']['batch_size'],
            num_workers=cfg['training']['num_workers'],
            fruit=fruit,
        )
        class_names = get_class_names(cfg['data']['data_dir'], fruit=fruit)
        print(f"  Classes: {class_names}")

        # Class weights for imbalanced data
        class_wts = get_class_weights(cfg['data']['data_dir'], fruit=fruit).to(self.device)
        # For binary: use pos_weight = weight_of_positive / weight_of_negative
        pos_weight = class_wts[1] / class_wts[0] if len(class_wts) > 1 else torch.tensor(1.0)

        # Model
        model = get_model_by_name(
            arch_name,
            num_classes=1,
            pretrained=cfg['model']['pretrained'],
            freeze_backbone=cfg['model']['freeze_backbone'],
        ).to(self.device)

        criterion = nn.BCELoss()

        # ── Phase 1: Train head only (backbone frozen) ──
        print("\n--- Phase 1: Training classifier head ---")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay'],
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(cfg['training']['num_epochs']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_metrics, _, _, _ = evaluate_model(model, val_loader, self.device)
            val_acc = val_metrics['accuracy'] * 100
            val_loss = 1 - val_metrics['f1_score']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            print(f"  Epoch {epoch+1:2d}/{cfg['training']['num_epochs']} | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | "
                  f"Val Acc: {val_acc:.1f}% | Val F1: {val_metrics['f1_score']:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best
        model.load_state_dict(best_state)

        # ── Phase 2: Fine-tune (unfreeze backbone) ──
        print("\n--- Phase 2: Fine-tuning backbone ---")
        model.unfreeze_backbone()
        optimizer_ft = optim.Adam(
            model.parameters(),
            lr=cfg['training']['fine_tune_lr'],
            weight_decay=cfg['training']['weight_decay'],
        )
        scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=3)
        patience_counter = 0

        for epoch in range(cfg['training']['fine_tune_epochs']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer_ft)
            val_metrics, _, _, _ = evaluate_model(model, val_loader, self.device)
            val_acc = val_metrics['accuracy'] * 100
            val_loss = 1 - val_metrics['f1_score']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            scheduler_ft.step(val_loss)

            print(f"  FT Epoch {epoch+1:2d}/{cfg['training']['fine_tune_epochs']} | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | "
                  f"Val Acc: {val_acc:.1f}% | Val F1: {val_metrics['f1_score']:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  Early stopping at FT epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)

        # Save model
        fruit_dir = os.path.join(self.save_dir, fruit)
        os.makedirs(fruit_dir, exist_ok=True)
        model_path = os.path.join(fruit_dir, f"{arch_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': arch_name,
            'fruit': fruit,
            'val_acc': best_val_acc,
            'class_names': class_names,
            'history': history,
        }, model_path)
        print(f"  Saved: {model_path} (val_acc={best_val_acc:.2f}%)")

        # Save training plot
        plot_path = os.path.join(fruit_dir, f"{arch_name}_history.png")
        plot_training_history(history, save_path=plot_path)

        return best_val_acc

    def train_all(self):
        """Train all 15 models (5 architectures x 3 fruits)."""
        results = {}

        for fruit in self.fruits:
            print(f"\n{'#'*60}")
            print(f"  FRUIT: {fruit.upper()}")
            print(f"{'#'*60}")

            fruit_results = {}
            for arch in self.arch_names:
                try:
                    val_acc = self._train_model(fruit, arch)
                    fruit_results[arch] = val_acc
                except Exception as e:
                    print(f"  ERROR training {arch} for {fruit}: {e}")
                    fruit_results[arch] = 0.0

                # Free GPU memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            results[fruit] = fruit_results

            # Select & copy best model for this fruit
            if fruit_results:
                best_arch = max(fruit_results, key=fruit_results.get)
                best_acc = fruit_results[best_arch]
                print(f"\n  BEST for {fruit}: {ARCHITECTURES[best_arch][1]} (val_acc={best_acc:.2f}%)")

                src = os.path.join(self.save_dir, fruit, f"{best_arch}.pth")
                dst = os.path.join(self.save_dir, fruit, f"{fruit}_best.pth")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        # Save summary
        summary_path = os.path.join(self.save_dir, "training_results.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print table
        print(f"\n{'='*60}")
        print("  TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"{'Fruit':<12} {'Architecture':<18} {'Val Accuracy':<12}")
        print("-" * 42)
        for fruit, archs in results.items():
            for arch, acc in archs.items():
                best = " <-- BEST" if acc == max(archs.values()) and acc > 0 else ""
                display = ARCHITECTURES[arch][1]
                print(f"{fruit:<12} {display:<18} {acc:<12.2f}{best}")
            print("-" * 42)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train fruit freshness models')
    parser.add_argument('--config', type=str, default='training/config.yaml')
    parser.add_argument('--fruit', type=str, default=None,
                        help='Train for a specific fruit only (apple/banana/strawberry)')
    parser.add_argument('--arch', type=str, default=None,
                        help='Train a specific architecture only (mobilenet/resnet/efficientnet/vgg/densenet)')
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)

    if args.fruit and args.arch:
        # Train single model
        trainer._train_model(args.fruit, args.arch)
    elif args.fruit:
        # Train all architectures for one fruit
        orig_fruits = trainer.fruits
        trainer.fruits = [args.fruit]
        trainer.train_all()
        trainer.fruits = orig_fruits
    else:
        # Train everything
        trainer.train_all()


if __name__ == '__main__':
    main()
