"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This script creates dummy model files for testing the benchmark script.
It creates a dummy decision tree model and a dummy BiLSTM model.
"""

import os
import joblib
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import argparse

# Define classes at module level to make them picklable
class DummyVectorizer:
    def transform(self, texts):
        return np.random.rand(len(texts), 10)

class DummyTokenizer:
    def __init__(self):
        self.word_index = {'hello': 1, 'world': 2}
        self.fitted = True
    
    def texts_to_sequences(self, texts):
        return [[1, 2] for _ in texts]

def create_dummy_decision_tree():
    """
    Create a dummy decision tree model
    
    Returns:
        dict: Dummy model data
    """
    # Create a dummy decision tree classifier
    model = DecisionTreeClassifier(max_depth=3)
    
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Fit model
    model.fit(X, y)
    
    # Create dummy vectorizer
    vectorizer = DummyVectorizer()
    
    # Create dummy config
    config = {
        'max_depth': 3,
        'min_samples_split': 2,
        'max_features': 0.5,
        'tfidf_max_features': 5000
    }
    
    # Create dummy model data
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'config': config
    }
    
    return model_data

def create_dummy_bilstm():
    """
    Create a dummy BiLSTM model
    
    Returns:
        dict: Dummy model data
    """
    # Create dummy config
    config = {
        'vocab_size': 10000,
        'embedding_dim': 100,
        'lstm_units': 64,
        'n_layers': 1,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }
    
    # Create dummy tokenizer
    tokenizer = DummyTokenizer()
    
    # Create dummy model state dict
    model_state_dict = {
        'embedding.weight': torch.rand(10000, 100),
        'rnn.weight_ih_l0': torch.rand(256, 100),
        'rnn.weight_hh_l0': torch.rand(256, 64),
        'rnn.bias_ih_l0': torch.rand(256),
        'rnn.bias_hh_l0': torch.rand(256),
        'fc.weight': torch.rand(3, 128),
        'fc.bias': torch.rand(3)
    }
    
    # Create dummy label maps
    label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    label_map_inv = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
    
    # Create dummy model data
    model_data = {
        'model_state_dict': model_state_dict,
        'config': config,
        'tokenizer': tokenizer,
        'label_map': label_map,
        'label_map_inv': label_map_inv
    }
    
    return model_data

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create dummy model files for testing')
    
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save dummy model files')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(args.output_dir, 'decision_tree'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bilstm'), exist_ok=True)
    
    # Create dummy decision tree model
    dt_model = create_dummy_decision_tree()
    dt_path = os.path.join(args.output_dir, 'decision_tree', 'dummy_decision_tree.pkl')
    
    try:
        joblib.dump(dt_model, dt_path)
        print(f"Created dummy decision tree model at {dt_path}")
    except Exception as e:
        print(f"Error saving decision tree model: {e}")
        
        # Try a simpler approach - create a very simple model
        simple_model = DecisionTreeClassifier(max_depth=3)
        simple_model.fit(np.random.rand(10, 5), np.random.randint(0, 3, 10))
        simple_dt_model = {
            'model': simple_model,
            'config': {'max_depth': 3}
        }
        joblib.dump(simple_dt_model, dt_path)
        print(f"Created simplified dummy decision tree model at {dt_path}")
    
    # Create dummy BiLSTM model
    bilstm_model = create_dummy_bilstm()
    bilstm_path = os.path.join(args.output_dir, 'bilstm', 'dummy_bilstm.pkl')
    
    try:
        torch.save(bilstm_model, bilstm_path)
        print(f"Created dummy BiLSTM model at {bilstm_path}")
    except Exception as e:
        print(f"Error saving BiLSTM model: {e}")
        
        # Try a simpler approach with minimal data
        simple_bilstm_model = {
            'config': {'vocab_size': 1000},
            'label_map': {'neutral': 0, 'entailment': 1, 'contradiction': 2},
            'label_map_inv': {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        }
        torch.save(simple_bilstm_model, bilstm_path)
        print(f"Created simplified dummy BiLSTM model at {bilstm_path}")

if __name__ == "__main__":
    main()
