"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This script implements a cascade training of models for NLI.
It uses a decision model (Md) to determine if a sample is entailment or contradiction,
then uses a positive model (Mp) or negative model (Mn) for the final prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the CascadeModel class
from cascade_model import CascadeModel

def load_data(dataset):
    """
    Load data from CSV file
    
    Args:
        dataset (str): Name of the dataset (train, dev, test)
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading {dataset} data...")
    data = pd.read_csv(f'data/{dataset}.csv')
    print(f"Dataset shape: {data.shape}")
    return data

def train_cascade_model(train_data, config):
    """
    Train the cascade model
    
    Args:
        train_data (pandas.DataFrame): Training data
        config (dict): Model configuration
        
    Returns:
        CascadeModel: Trained cascade model
    """
    print("Training cascade model...")
    
    # Create the cascade model
    model = CascadeModel(config)
    
    # Train the model
    model.train_model(train_data)
    
    return model

def evaluate_model(model, test_data):
    """
    Evaluate the model on test data
    
    Args:
        model (CascadeModel): Trained cascade model
        test_data (pandas.DataFrame): Test data
        
    Returns:
        dict: Evaluation results
    """
    print("Evaluating cascade model...")
    
    # Evaluate the model
    results = model.evaluate(test_data)
    
    # Print results
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    for label, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
    
    return results

def save_model(model, model_dir='models/cascade'):
    """
    Save the model to disk
    
    Args:
        model (CascadeModel): Trained cascade model
        model_dir (str): Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    print(f"Saving model to {model_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a descriptive filename based on model parameters
    param_str = f"max_depth={model.max_depth}_min_samples_split={model.min_samples_split}_tfidf_max_features={model.max_features}"
    model_path = f"{model_dir}/model_cascade_{param_str}.pkl"
    
    # Save the model
    model.save(model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def plot_confusion_matrix(cm, accuracy, save_path):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        accuracy (float): Model accuracy
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Use text labels instead of numeric indices
    label_names = ['neutral', 'entailment', 'contradiction']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_names,
               yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Include accuracy in the title
    plt.title(f'Cascade Confusion Matrix (Accuracy: {accuracy:.4f})')
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved confusion matrix to {save_path}")

def main():
    """Main function"""
    # Define model configuration
    config = {
        'max_depth': 10,
        'min_samples_split': 2,
        'tfidf_max_features': 5000
    }
    
    # Load and preprocess the data
    train_data = load_data("train")
    test_data = load_data("test")
    
    # Train the cascade model
    model = train_cascade_model(train_data, config)
    
    # Evaluate the model
    results = evaluate_model(model, test_data)
    
    # Save the model
    model_path = save_model(model)
    
    # Plot and save confusion matrix
    results_dir = 'results/cascade'
    os.makedirs(results_dir, exist_ok=True)
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['accuracy'],
        os.path.join(results_dir, 'cascade_confusion_matrix.png')
    )
    
    print("Cascade model implementation completed!")

if __name__ == "__main__":
    main()
