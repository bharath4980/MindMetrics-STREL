"""
Model Comparison and Evaluation
Compares multiple classification models using accuracy, precision, recall, F1-score
and ROC curves
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

class ModelEvaluator:
    """Collects and compares metrics from multiple classification models"""
    
    def __init__(self):
        self.metrics = []
        self.roc_data = []
    
    def add_model(self, model_name, accuracy, precision, recall, f1_score, fpr=None, tpr=None):
        """
        Add a model's evaluation metrics
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            fpr: False positive rates for ROC curve (optional)
            tpr: True positive rates for ROC curve (optional)
        """
        self.metrics.append({
            'Model': model_name, 
            'Accuracy': accuracy, 
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })
        if fpr is not None and tpr is not None:
            self.roc_data.append({'model': model_name, 'fpr': fpr, 'tpr': tpr})
    
    def print_summary(self):
        """Print summary table of all model metrics"""
        df = pd.DataFrame(self.metrics)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    def plot_metrics(self, save_path=None):
        """
        Plot bar charts comparing all metrics across models
        
        Args:
            save_path: Path to save the plot (optional)
        """
        df = pd.DataFrame(self.metrics)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['steelblue', 'coral', 'green', 'purple']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            ax.bar(df['Model'], df[metric], color=color, alpha=0.8)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(df[metric]):
                ax.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nMetrics saved to: {save_path}")
        plt.show()
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all models on the same graph
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.roc_data:
            print("\nNo ROC data available")
            return
        
        plt.figure(figsize=(10, 8))
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
        
        for i, data in enumerate(self.roc_data):
            plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=2,
                    label=f"{data['model']}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        plt.show()
    
    def save_metrics(self, filepath):
        """
        Save metrics to CSV file
        
        Args:
            filepath: Path to save the CSV file
        """
        pd.DataFrame(self.metrics).to_csv(filepath, index=False)
        print(f"Metrics saved to: {filepath}")


# ─────────────────────────────────────────────
# COMPARE MODELS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    evaluator = ModelEvaluator()
    
    # XGBoost
    from xg_boost_model import train_and_evaluate_xgboost
    result = train_and_evaluate_xgboost()
    evaluator.add_model(
        result['model_name'], 
        result['accuracy'], 
        result['precision'],
        result['recall'],
        result['f1_score'],
        fpr=result['fpr'], 
        tpr=result['tpr']
    )
    
    # TODO: Add other models
    # from logistic_regression_model import train_and_evaluate_lr
    # result = train_and_evaluate_lr()
    # evaluator.add_model(result['model_name'], result['accuracy'], result['precision'],
    #                    result['recall'], result['f1_score'], fpr=result['fpr'], tpr=result['tpr'])
    
    evaluator.print_summary()
    evaluator.plot_metrics(save_path="../../results/metrics/model_comparison.png")
    evaluator.plot_roc_curves(save_path="../../results/metrics/roc_curves.png")
    evaluator.save_metrics("../../results/metrics/model_metrics.csv")
