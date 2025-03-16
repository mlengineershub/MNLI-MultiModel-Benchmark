"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This module implements the training and evaluation process for NLI models.
It includes:
- Loading and preprocessing data
- Hyperparameter search
- Model training and evaluation
- Confusion matrix visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import itertools
import time
import torch
from sklearn.metrics import confusion_matrix, classification_report
from models import create_model
from preprocessing import TfidfModel

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(train_path, dev_path):
    """
    Load data from CSV files
    
    Args:
        train_path (str): Path to the training data
        dev_path (str): Path to the development data
        
    Returns:
        tuple: Training and development dataframes
    """
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    
    # Check for NaN values and drop them
    train_df = train_df.dropna()
    dev_df = dev_df.dropna()
    
    return train_df, dev_df

def preprocess_data(train_df, dev_df):
    """
    Preprocess the data for training and evaluation
    
    Args:
        train_df (pandas.DataFrame): Training data
        dev_df (pandas.DataFrame): Development data
        
    Returns:
        tuple: Preprocessed training and development data
    """
    # For now, we'll just return the dataframes as they are
    # The actual preprocessing will be done by the model classes
    return train_df, dev_df

def generate_hyperparameter_combinations(model_config):
    """
    Generate all combinations of hyperparameters
    
    Args:
        model_config (dict): Model configuration with hyperparameter lists
        
    Returns:
        list: List of hyperparameter dictionaries
    """
    # Extract hyperparameter lists
    param_lists = {}
    for key, value in model_config.items():
        if isinstance(value, list):
            param_lists[key] = value
    
    # Generate all combinations
    keys = list(param_lists.keys())
    values = list(param_lists.values())
    combinations = list(itertools.product(*values))
    
    # Convert to list of dictionaries
    result = []
    for combo in combinations:
        param_dict = {key: value for key, value in zip(keys, combo)}
        # Add non-list parameters
        for key, value in model_config.items():
            if not isinstance(value, list):
                param_dict[key] = value
        result.append(param_dict)
    
    return result

def train_and_evaluate_model(model_name, model_config, train_df, dev_df, common_config):
    """
    Train and evaluate a model with hyperparameter search
    
    Args:
        model_name (str): Name of the model
        model_config (dict): Model configuration
        train_df (pandas.DataFrame): Training data
        dev_df (pandas.DataFrame): Development data
        common_config (dict): Common configuration parameters
        
    Returns:
        tuple: Best model, best hyperparameters, and evaluation results
    """
    print(f"Training and evaluating {model_name}...")
    
    # Generate hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations(model_config)
    print(f"Generated {len(hyperparameter_combinations)} hyperparameter combinations")
    
    # Initialize variables to track best model
    best_model = None
    best_params = None
    best_accuracy = 0.0
    best_results = None
    
    # Train and evaluate each combination
    for i, params in enumerate(hyperparameter_combinations):
        print(f"Combination {i+1}/{len(hyperparameter_combinations)}: {params}")
        
        # Create model with current hyperparameters
        model_params = {**common_config, **params}
        
        # Create model based on model name
        if model_name == 'tfidf':
            model = TfidfModel(model_params)
        else:
            model = create_model(model_name, model_params)
        
        # Train model
        start_time = time.time()
        history = model.train_model(
            train_df, 
            dev_df=None,  # Use validation_split instead
            epochs=params.get('epochs', 10),
            batch_size=common_config.get('batch_size', 64),
            validation_split=common_config.get('validation_split', 0.1)
        )
        train_time = time.time() - start_time
        
        # Evaluate model on dev set
        results = model.evaluate(dev_df)
        accuracy = results['accuracy']
        
        print(f"Accuracy: {accuracy:.4f}, Training time: {train_time:.2f}s")
        
        # Update best model if current is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
            best_results = results
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            model.save(f"models/{model_name}_best.pt")
    
    print(f"Best {model_name} model:")
    print(f"Parameters: {best_params}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    return best_model, best_params, best_results

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): Class names
        title (str): Plot title
        cmap: Colormap
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{title.lower().replace(' ', '_')}.png")
    plt.close()

def main():
    """Main function to run the training and evaluation process"""
    # Start timer
    start_time = time.time()
    
    # Load configuration
    config = load_config('config/configuration.yaml')
    common_config = config.get('common', {})
    
    # Load data
    print("Loading data...")
    train_df, dev_df = load_data('data/train.csv', 'data/dev1.csv')
    print(f"Loaded {len(train_df)} training samples and {len(dev_df)} development samples")
    
    # Preprocess data
    print("Preprocessing data...")
    train_df, dev_df = preprocess_data(train_df, dev_df)
    
    # Train and evaluate models
    results = {}
    
    # List of models to train and evaluate
    models_to_train = ['naive_bayes']
    
    for model_name in models_to_train:
        if model_name in config:
            model_config = config[model_name]
            best_model, best_params, best_results = train_and_evaluate_model(
                model_name, model_config, train_df, dev_df, common_config
            )
            
            # Store results
            results[model_name] = {
                'params': best_params,
                'accuracy': best_results['accuracy'],
                'confusion_matrix': best_results['confusion_matrix'],
                'classification_report': best_results['classification_report']
            }
            
            # Plot confusion matrix
            plot_confusion_matrix(
                best_results['confusion_matrix'],
                classes=['neutral', 'entailment', 'contradiction'],
                title=f'{model_name.capitalize()} Confusion Matrix'
            )
    
    # Print summary of results
    print("\nSummary of results:")
    for model_name, model_results in results.items():
        print(f"{model_name.capitalize()}:")
        print(f"  Accuracy: {model_results['accuracy']:.4f}")
        print(f"  Best parameters: {model_results['params']}")
        print(f"  Classification report:")
        for class_name, metrics in model_results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"    {class_name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")
        print()
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
