"""
Evaluation Script for Fruit Freshness Detection
Evaluates all 15 models (5 architectures x 3 fruits) on test sets
Generates: confusion matrices, comparison charts, metrics JSON
"""

import torch
import yaml
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import get_loaders, get_class_names
from utils.metrics import evaluate_model, print_classification_report, plot_roc_curve
from utils.visualization import plot_confusion_matrix, visualize_predictions
from models import get_model_by_name, ARCHITECTURES


class Evaluator:
    """Evaluates all trained fruit-freshness models"""

    def __init__(self, config_path='training/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fruits = self.config['fruits']
        self.arch_names = self.config['architectures']
        self.save_dir = self.config['training']['save_dir']

    def _load_model(self, fruit, arch_name):
        """Load a trained model."""
        model_path = os.path.join(self.save_dir, fruit, f"{arch_name}.pth")
        if not os.path.exists(model_path):
            return None

        model = get_model_by_name(arch_name, num_classes=1, pretrained=False, freeze_backbone=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def evaluate_single(self, fruit, arch_name, save_dir):
        """Evaluate a single model."""
        model = self._load_model(fruit, arch_name)
        if model is None:
            print(f"  Skipping {arch_name}: model not found")
            return None

        _, _, test_loader = get_loaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            fruit=fruit,
        )
        class_names = get_class_names(self.config['data']['data_dir'], fruit=fruit)

        metrics, y_true, y_pred, y_proba = evaluate_model(model, test_loader, self.device)

        display_name = ARCHITECTURES[arch_name][1]
        print(f"  {display_name:18s} | Acc: {metrics['accuracy']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1_score']:.4f}")

        # Confusion matrix
        cm_path = os.path.join(save_dir, f"{fruit}_{arch_name}_cm.png")
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)

        # ROC curve
        if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
            roc_path = os.path.join(save_dir, f"{fruit}_{arch_name}_roc.png")
            plot_roc_curve(y_true, y_proba, save_path=roc_path)

        return {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc', 0.0),
            'confusion_matrix': metrics['confusion_matrix'],
        }

    def plot_comparisons(self, results, save_dir):
        """Generate comparison bar charts."""
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

        for fruit in self.fruits:
            if fruit not in results:
                continue

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"{fruit.capitalize()} - Architecture Comparison", fontsize=14)

            for idx, metric in enumerate(metric_names):
                archs = []
                values = []
                for arch in self.arch_names:
                    if arch in results[fruit]:
                        archs.append(ARCHITECTURES[arch][1])
                        values.append(results[fruit][arch][metric])

                colors = plt.cm.Set2(np.linspace(0, 1, len(archs)))
                bars = axes[idx].bar(archs, values, color=colors)
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].set_ylim(0, 1.05)
                axes[idx].tick_params(axis='x', rotation=45)
                for bar, val in zip(bars, values):
                    axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{fruit}_comparison.png"), dpi=150)
            plt.close()

        # Best model summary
        fig, ax = plt.subplots(figsize=(10, 6))
        fruit_labels, best_f1s, best_names = [], [], []

        for fruit in self.fruits:
            if fruit not in results or not results[fruit]:
                continue
            best_arch = max(results[fruit], key=lambda a: results[fruit][a]['f1_score'])
            fruit_labels.append(fruit.capitalize())
            best_f1s.append(results[fruit][best_arch]['f1_score'])
            best_names.append(ARCHITECTURES[best_arch][1])

        colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
        bars = ax.bar(fruit_labels, best_f1s, color=colors[:len(fruit_labels)])
        for bar, name, val in zip(bars, best_names, best_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{name}\n{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_title('Best Model per Fruit (F1-Score)')
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_models_summary.png"), dpi=150)
        plt.close()

    def evaluate_all(self, save_dir='evaluation/results'):
        """Evaluate all 15 models."""
        os.makedirs(save_dir, exist_ok=True)
        all_results = {}

        for fruit in self.fruits:
            print(f"\n{'='*60}")
            print(f"  Evaluating {fruit.upper()}")
            print(f"{'='*60}")

            fruit_results = {}
            for arch in self.arch_names:
                result = self.evaluate_single(fruit, arch, save_dir)
                if result:
                    fruit_results[arch] = result

            all_results[fruit] = fruit_results

            if fruit_results:
                best = max(fruit_results, key=lambda a: fruit_results[a]['f1_score'])
                print(f"\n  BEST for {fruit}: {ARCHITECTURES[best][1]} "
                      f"(F1={fruit_results[best]['f1_score']:.4f})")

        # Save results
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Comparison plots
        self.plot_comparisons(all_results, save_dir)

        # Final summary table
        print(f"\n{'='*70}")
        print("  FINAL EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"{'Fruit':<12} {'Architecture':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 62)
        for fruit in self.fruits:
            if fruit not in all_results:
                continue
            for arch in self.arch_names:
                if arch not in all_results[fruit]:
                    continue
                m = all_results[fruit][arch]
                best_marker = ""
                best_arch = max(all_results[fruit], key=lambda a: all_results[fruit][a]['f1_score'])
                if arch == best_arch:
                    best_marker = " *"
                display = ARCHITECTURES[arch][1]
                print(f"{fruit:<12} {display:<18} {m['accuracy']:<8.4f} {m['precision']:<8.4f} "
                      f"{m['recall']:<8.4f} {m['f1_score']:<8.4f}{best_marker}")
            print("-" * 62)
        print("(* = best model for this fruit)")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate all freshness models')
    parser.add_argument('--config', type=str, default='training/config.yaml')
    parser.add_argument('--save_dir', type=str, default='evaluation/results')
    args = parser.parse_args()

    evaluator = Evaluator(args.config)
    evaluator.evaluate_all(save_dir=args.save_dir)


if __name__ == '__main__':
    main()
