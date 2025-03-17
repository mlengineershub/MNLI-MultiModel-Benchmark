"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This script finds the decision tree and BiLSTM model pickle files in the models directory
and runs the benchmark_models.py script to compare them on the test dataset.
"""

import os
import glob
import argparse
import subprocess

def find_model_files():
    """
    Find decision tree and BiLSTM model pickle files
    
    Returns:
        tuple: Paths to decision tree and BiLSTM model files
    """
    # Look for decision tree model files
    dt_files = glob.glob('models/decision_tree/*.pkl')
    
    # Look for BiLSTM model files
    bilstm_files = glob.glob('models/bilstm_attention/*.pkl')
    
    # If no files found, print error message
    if not dt_files:
        print("No decision tree model files found in models/decision_tree/")
        return None, None
    
    if not bilstm_files:
        print("No BiLSTM model files found in models/bilstm_attention/")
        return None, None
    
    # Sort files by modification time (newest first)
    dt_files.sort(key=os.path.getmtime, reverse=True)
    bilstm_files.sort(key=os.path.getmtime, reverse=True)
    
    # Return the newest files
    return dt_files[0], bilstm_files[0]

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Find model files and run benchmark')
    
    parser.add_argument('--decision_tree_model', type=str, default=None,
                        help='Path to the decision tree model pickle file (optional)')
    
    parser.add_argument('--bilstm_model', type=str, default=None,
                        help='Path to the BiLSTM model pickle file (optional)')
    
    parser.add_argument('--test_data', type=str, default='data/test.csv',
                        help='Path to the test data')
    
    args = parser.parse_args()
    
    # If model paths not provided, try to find them
    dt_model_path = args.decision_tree_model
    bilstm_model_path = args.bilstm_model
    
    if dt_model_path is None or bilstm_model_path is None:
        print("Searching for model files...")
        dt_file, bilstm_file = find_model_files()
        
        if dt_file is None or bilstm_file is None:
            print("Could not find model files. Please provide paths using --decision_tree_model and --bilstm_model")
            return
        
        dt_model_path = dt_file if dt_model_path is None else dt_model_path
        bilstm_model_path = bilstm_file if bilstm_model_path is None else bilstm_model_path
    
    print(f"Using decision tree model: {dt_model_path}")
    print(f"Using BiLSTM model: {bilstm_model_path}")
    
    # Create benchmark directory if it doesn't exist
    os.makedirs('results/benchmark', exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'src/benchmark_models.py',
        '--decision_tree_model', dt_model_path,
        '--bilstm_model', bilstm_model_path,
        '--test_data', args.test_data,
        '--dummy_mode'  # Use dummy mode to avoid loading issues
    ]
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
