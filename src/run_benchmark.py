"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This script is a simple wrapper to run the benchmark_models.py script with
the specified decision tree and BiLSTM model pickle files.
"""

import os
import argparse
import subprocess

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run benchmark for Decision Tree and BiLSTM models')
    
    parser.add_argument('--decision_tree_model', type=str, required=True,
                        help='Path to the decision tree model pickle file')
    
    parser.add_argument('--bilstm_model', type=str, required=True,
                        help='Path to the BiLSTM model pickle file')
    
    parser.add_argument('--test_data', type=str, default='data/test.csv',
                        help='Path to the test data')
    
    args = parser.parse_args()
    
    # Create benchmark directory if it doesn't exist
    os.makedirs('results/benchmark', exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'src/benchmark_models.py',
        '--decision_tree_model', args.decision_tree_model,
        '--bilstm_model', args.bilstm_model,
        '--test_data', args.test_data
    ]
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
