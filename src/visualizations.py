import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def plot_model_performance(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """
    Plot model performance metrics (MSE and R²) for each target variable.
    
    Args:
        results: Dictionary containing performance metrics for each target
        save_path: Optional path to save the plot
    """
    targets = list(results.keys())
    metrics = ['test_mse', 'test_r2']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Metrics by Target Variable', fontsize=16)
    
    # Plot MSE
    mse_values = [results[t]['test_mse'] for t in targets]
    sns.barplot(x=targets, y=mse_values, ax=axes[0])
    axes[0].set_title('Mean Squared Error (MSE)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].set_ylabel('MSE')
    
    # Plot R²
    r2_values = [results[t]['test_r2'] for t in targets]
    sns.barplot(x=targets, y=r2_values, ax=axes[1])
    axes[1].set_title('R² Score')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    axes[1].set_ylabel('R²')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Performance plot saved to {save_path}")
    else:
        plt.show()

def plot_prediction_radar(predictions: Dict[str, float], save_path: Optional[str] = None):
    """
    Create a radar plot of predicted leadership competencies.
    
    Args:
        predictions: Dictionary of competency predictions
        save_path: Optional path to save the plot
    """
    # Number of variables
    N = len(predictions)
    
    # Compute angle for each competency
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Get values and add first value again to close the loop
    values = list(predictions.values())
    values += values[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.replace('_', ' ').title() for k in predictions.keys()])
    
    # Set title
    plt.title('Leadership Competency Predictions', size=15, y=1.1)
    
    # Set y-axis limits
    ax.set_ylim(0, 5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Radar plot saved to {save_path}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names: List[str], save_path: Optional[str] = None):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Optional path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for metric, values in history.items():
        plt.plot(values, label=metric)
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show() 