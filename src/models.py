import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import time

class Tokenizer:
    """Simple tokenizer for text data"""
    
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        self.document_count = 0
    
    def fit_on_texts(self, texts):
        for text in texts:
            self.document_count += 1
            for word in text.split():
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        if self.num_words:
            sorted_words = sorted_words[:self.num_words]
        
        for i, (word, _) in enumerate(sorted_words):
            self.word_index[word] = i + 1
            self.index_word[i + 1] = word
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                if word in self.word_index:
                    # Ensure the index is within the vocabulary size
                    idx = self.word_index[word]
                    if self.num_words is None or idx < self.num_words:
                        sequence.append(idx)
            sequences.append(sequence)
        return sequences

    def add_special_tokens(self, special_tokens):
        """
        Add special tokens to the tokenizer vocabulary
        
        Args:
            special_tokens (dict): Dictionary of special tokens to add
        """
        if not self.num_words:
            # If no vocabulary size limit, just add tokens
            max_index = max(self.word_index.values()) if self.word_index else 0
            for i, (token_name, token) in enumerate(special_tokens.items()):
                idx = max_index + i + 1
                self.word_index[token] = idx
                self.index_word[idx] = token
                self.word_counts[token] = float('inf')
            return
        
        # Calculate how many words we need to remove to make space
        num_to_remove = len(special_tokens) - (self.num_words - len(self.word_index))
        
        if num_to_remove > 0:
            # Remove least frequent words to make space
            sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1])
            words_to_remove = [word for word, count in sorted_words[:num_to_remove] 
                             if word not in special_tokens.values()]
            
            for word in words_to_remove:
                if word in self.word_index:
                    idx = self.word_index.pop(word)
                    self.index_word.pop(idx)
                    self.word_counts.pop(word)
        
        # Add special tokens
        for token_name, token in special_tokens.items():
            if token not in self.word_index:
                # Find first available index
                idx = next(i for i in range(1, self.num_words + 1) 
                          if i not in self.index_word)
                
                self.word_index[token] = idx
                self.index_word[idx] = token
                self.word_counts[token] = float('inf')

class DecisionTreeModel:
    """Decision Tree model for NLI with MLflow tracking"""
    
    def __init__(self, config):
        self.config = config
        self.model = DecisionTreeClassifier(
            max_depth=config.get('max_depth', 10),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
        )
        
        tfidf_max_features = 5000
        if 'tfidf_max_features' in config:
            tfidf_max_features = config['tfidf_max_features']
        
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2)
        )
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        
    def preprocess_data(self, df):
        # Combine text and assertion
        combined_text = df['text'] + " " + df['assertion']
        
        # Vectorize text
        if not hasattr(self, 'fitted_vectorizer'):
            X = self.vectorizer.fit_transform(combined_text)
            self.fitted_vectorizer = self.vectorizer
        else:
            X = self.fitted_vectorizer.transform(combined_text)
            
        # Convert labels
        y = df['label'].map(self.label_map).values
        
        return X, y
    
    def train_model(self, train_df, dev_df=None, epochs=None, batch_size=None, validation_split=None):
        """
        Train the model
        
        Args:
            train_df (pandas.DataFrame): Training data
            dev_df (pandas.DataFrame, optional): Development data
            epochs (int, optional): Not used for Decision Tree
            batch_size (int, optional): Not used for Decision Tree
            validation_split (float, optional): Not used for Decision Tree
            
        Returns:
            dict: Training history (empty for Decision Tree)
        """
        # Preprocess data
        X_train, y_train = self.preprocess_data(train_df)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_preds = self.model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        print(f"Training accuracy: {train_acc:.4f}")
        
        # Evaluate on dev data if provided
        if dev_df is not None:
            X_dev, y_dev = self.preprocess_data(dev_df)
            dev_preds = self.model.predict(X_dev)
            dev_acc = np.mean(dev_preds == y_dev)
            print(f"Development accuracy: {dev_acc:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_dev, dev_preds)
            
            # Create classification report
            report = classification_report(
                y_dev, dev_preds,
                target_names=list(self.label_map.keys()),
                output_dict=True
            )
        
        # Return empty history (not applicable for Decision Tree)
        return {}
            
    def evaluate(self, df):
        X, y_true = self.preprocess_data(df)
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.label_map.keys()),
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', save_dir=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.label_map.keys()),
                   yticklabels=list(self.label_map.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()
    
    def save(self, model_dir):
        """Save model components to disk"""
        # Extract model parameters for filename
        param_str = "_".join([f"{k}={v}" for k, v in self.config.items() if not isinstance(v, list)])
        
        # Create directory
        save_dir = os.path.dirname(model_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract filename from model_dir
        filename = os.path.basename(model_dir)
        
        # Save model with descriptive filename
        model_path = model_dir
        joblib.dump({
            'model': self.model,
            'vectorizer': self.fitted_vectorizer,
            'config': self.config
        }, model_path)
        
        return save_dir
    
    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data['model']
        self.fitted_vectorizer = data['vectorizer']
        self.config = data['config']

class BiLSTMAttention(nn.Module):
    """Bi-LSTM with Attention model for NLI"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# Factory function to create models
def create_model(model_name, config):
    if model_name == 'decision_tree':
        return DecisionTreeModel(config)
    elif model_name == 'bilstm_attention':
        # Import here to avoid circular imports
        from bilstm_attention_model import BiLSTMAttentionModel
        return BiLSTMAttentionModel(config)
    elif model_name == 'transformer':
        # Transformer model would be implemented here
        raise NotImplementedError("Transformer model not implemented yet")
    elif model_name == 'bert':
        # BERT model would be implemented here
        raise NotImplementedError("BERT model not implemented yet")
    else:
        raise ValueError(f"Model {model_name} not supported")