"""
@Authors: Ilyes DJERFAF, Nazim KESKES

Main script for running the NLI model training and evaluation.
This script provides a command-line interface for running the code.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from train_evaluate import main as train_evaluate_main

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='NLI Model Training and Evaluation')
    
    parser.add_argument('--train', action='store_true',
                        help='Train and evaluate models')
    
    parser.add_argument('--config', type=str, default='config/configuration.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--train_data', type=str, default='data/train.csv',
                        help='Path to training data')
    
    parser.add_argument('--dev_data', type=str, default='data/dev1.csv',
                        help='Path to development data')
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=['naive_bayes'],
                        help='Models to train and evaluate')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def check_files_exist(files):
    """Check if files exist"""
    for file in files:
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist")
            return False
    return True

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Check if required files exist
    required_files = [args.config, args.train_data, args.dev_data]
    if not check_files_exist(required_files):
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training and evaluation
    if args.train:
        print("Running training and evaluation...")
        train_evaluate_main()
    else:
        print("No action specified. Use --train to train and evaluate models.")
        print("For more options, use --help.")

if __name__ == "__main__":
    main()
