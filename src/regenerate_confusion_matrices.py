"""
Script to regenerate confusion matrices with text labels instead of numeric indices.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(dev_path):
    """Load development data"""
    dev_df = pd.read_csv(dev_path)
    dev_df = dev_df.dropna()
    return dev_df

def main():
    # Load configuration
    config = load_config('config/configuration.yaml')
    
    # Load development data
    dev_df = load_data('data/dev1.csv')
    
    # Load results CSV
    results_df = pd.read_csv('results/decision_tree/all_results.csv')
    
    # Define label names
    label_names = ['neutral', 'entailment', 'contradiction']
    label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    
    # Process each model
    for _, row in results_df.iterrows():
        model_file = row['model_file']
        cm_file = row['confusion_matrix']
        accuracy = row['accuracy']
        
        print(f"Processing {model_file}...")
        
        # Load model
        model_data = joblib.load(model_file)
        
        # Extract model components
        if isinstance(model_data, dict):
            # New format
            model = model_data['model']
            vectorizer = model_data['vectorizer']
        else:
            # Old format
            model = model_data
            vectorizer = None
        
        # Prepare data for prediction
        combined_text = dev_df['text'] + " " + dev_df['assertion']
        X = vectorizer.transform(combined_text)
        y_true = dev_df['label'].map(label_map).values
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix with text labels
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names,
                   yticklabels=label_names)
        
        # Extract model name and parameters from filename
        model_name = os.path.basename(model_file).split('_')[1]
        plt.title(f'{model_name.capitalize()} Confusion Matrix (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save the figure
        plt.savefig(cm_file)
        plt.close()
        
        print(f"Saved confusion matrix to {cm_file}")

if __name__ == "__main__":
    main()
