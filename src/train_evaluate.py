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
    
    # Create directories for this model type
    model_dir = f"models/{model_name}"
    results_dir = f"results/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations(model_config)
    print(f"Generated {len(hyperparameter_combinations)} hyperparameter combinations")
    
    # Initialize variables to track best model
    best_model = None
    best_params = None
    best_accuracy = 0.0
    best_results = None
    
    # Initialize list to store all results
    all_results = []
    
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
        
        # Create descriptive filename based on model parameters
        param_str = "_".join([f"{k}={v}" for k, v in params.items() if not isinstance(v, list)])
        
        # Save model with descriptive filename
        model_filename = f"{model_dir}/model_{model_name}_{param_str}.pkl"
        model.save(model_filename)
        
        # Save confusion matrix with descriptive filename
        cm_filename = f"{results_dir}/confusion_matrix_{model_name}_{param_str}.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name.capitalize()} Confusion Matrix (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_filename)
        plt.close()
        
        # Store results with file paths
        result_entry = {
            'params': params,
            'accuracy': accuracy,
            'model_file': model_filename,
            'confusion_matrix': cm_filename,
            'training_time': train_time
        }
        all_results.append(result_entry)
        
        # Update best model if current is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = params
            best_results = results
            
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{results_dir}/all_results.csv", index=False)
    
    print(f"Best {model_name} model:")
    print(f"Parameters: {best_params}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    return best_model, best_params, best_results

def main(config_path='config/configuration.yaml', train_path='data/train.csv', 
         dev_path='data/dev1.csv', models_to_train=None, output_dir='results'):
    """
    Main function to run the training and evaluation process
    
    Args:
        config_path (str): Path to the configuration file
        train_path (str): Path to the training data
        dev_path (str): Path to the development data
        models_to_train (list): List of models to train and evaluate
        output_dir (str): Directory to save results
    """
    # Start timer
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    common_config = config.get('common', {})
    
    # Load data
    print("Loading data...")
    train_df, dev_df = load_data(train_path, dev_path)
    print(f"Loaded {len(train_df)} training samples and {len(dev_df)} development samples")
    
    # Preprocess data
    print("Preprocessing data...")
    train_df, dev_df = preprocess_data(train_df, dev_df)
    
    # Train and evaluate models
    results = {}
    
    # If models_to_train is not specified, use all models in the config file
    if models_to_train is None:
        models_to_train = list(config.keys())
        # Remove common section as it's not a model
        if 'common' in models_to_train:
            models_to_train.remove('common')
    
    print(f"Models to train: {', '.join(models_to_train)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
