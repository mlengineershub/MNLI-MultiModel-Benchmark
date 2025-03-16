"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This module implements a BiLSTM with Attention model for NLI.
It's a deep learning approach that uses bidirectional LSTM with attention mechanism.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from models import BiLSTMAttention, Tokenizer

class BiLSTMAttentionModel:
    """BiLSTM with Attention model for NLI"""
    
    def __init__(self, config):
        """
        Initialize the BiLSTM with Attention model
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        # Use CUDA if available
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set model parameters
        self.vocab_size = config.get('vocab_size', 20000)  # Increase vocabulary size
        self.embedding_dim = config.get('embedding_dim', 300)
        self.hidden_dim = config.get('lstm_units', 128)
        self.n_layers = config.get('n_layers', 1)
        self.dropout = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Create model
        self.model = BiLSTMAttention(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=3,  # 3 classes: neutral, entailment, contradiction
            n_layers=self.n_layers,
            bidirectional=True,
            dropout=self.dropout,
            pad_idx=0
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Create tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        
        # Define label mapping
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.label_map_inv = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
    
    def preprocess_data(self, df):
        """
        Preprocess data for training or evaluation
        
        Args:
            df (pandas.DataFrame): Data to preprocess
            
        Returns:
            tuple: Preprocessed data
        """
        # Add special token for "x" marker
        self.tokenizer.add_special_tokens({'x_marker': '[X]'})
        
        # Combine text and assertion with "x" marker
        texts = df['text'].tolist()
        assertions = df['assertion'].tolist()
        combined = [f"{text} [X] {assertion}" for text, assertion in zip(texts, assertions)]
        
        # Fit tokenizer on training data if not already fitted
        if not hasattr(self.tokenizer, 'fitted'):
            self.tokenizer.fit_on_texts(combined)
            self.tokenizer.fitted = True
        
        # Convert combined text + [X] + assertion to sequences
        combined_sequences = self.tokenizer.texts_to_sequences(combined)
        
        # Pad sequences
        max_len = self.config.get('max_sequence_length', 100)
        
        # Function to pad sequences
        def pad_sequences(sequences, max_len):
            padded = []
            lengths = []
            for seq in sequences:
                if len(seq) > max_len:
                    padded.append(seq[:max_len])
                    lengths.append(max_len)
                elif len(seq) == 0:
                    # Handle empty sequences by adding a single padding token
                    padded.append([0] * max_len)
                    lengths.append(max_len)  # Use max_len to ensure proper packing
                else:
                    padded.append(seq + [0] * (max_len - len(seq)))
                    lengths.append(len(seq))
            return padded, lengths
        
        # Pad combined sequences
        combined_padded, combined_lengths = pad_sequences(combined_sequences, max_len)
        
        # Convert to tensors
        combined_tensor = torch.LongTensor(combined_padded)
        combined_lengths_tensor = torch.LongTensor(combined_lengths)
        
        # Convert labels to indices
        labels = df['label'].map(self.label_map).values
        labels_tensor = torch.LongTensor(labels)
        
        return (combined_tensor, combined_lengths_tensor, labels_tensor)
    
    def train_model(self, train_df, dev_df=None, epochs=10, batch_size=64, validation_split=0.1):
        """
        Train the model
        
        Args:
            train_df (pandas.DataFrame): Training data
            dev_df (pandas.DataFrame, optional): Development data
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_split (float): Validation split if dev_df is None
            
        Returns:
            dict: Training history
        """
        # Preprocess data
        train_data = self.preprocess_data(train_df)
        
        # Create dataset
        train_dataset = TensorDataset(*train_data)
        
        # Use the provided batch_size
        config_batch_size = batch_size
        
        # Split into training and validation sets if dev_df is None
        if dev_df is None and validation_split > 0:
            val_size = int(len(train_dataset) * validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=config_batch_size)
        else:
            val_loader = None
            if dev_df is not None:
                val_data = self.preprocess_data(dev_df)
                val_dataset = TensorDataset(*val_data)
                val_loader = DataLoader(val_dataset, batch_size=config_batch_size)
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=config_batch_size, shuffle=True)
        
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None
        
        # Train model
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Get batch data
                combined, combined_lengths, labels = batch
                
                # Sort by sequence length in descending order for pack_padded_sequence
                combined_lengths, sorted_idx = combined_lengths.sort(descending=True)
                combined = combined[sorted_idx]
                labels = labels[sorted_idx]
                
                # Move to device
                combined = combined.to(self.device)
                combined_lengths = combined_lengths.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(combined, combined_lengths)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * combined.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate epoch statistics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        # Get batch data
                        combined, combined_lengths, labels = batch
                        
                        # Sort by sequence length in descending order for pack_padded_sequence
                        combined_lengths, sorted_idx = combined_lengths.sort(descending=True)
                        combined = combined[sorted_idx]
                        labels = labels[sorted_idx]
                        
                        # Move to device
                        combined = combined.to(self.device)
                        combined_lengths = combined_lengths.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(combined, combined_lengths)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)
                        
                        # Update statistics
                        val_loss += loss.item() * combined.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Calculate epoch statistics
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state_dict = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        self.model.load_state_dict(best_state_dict)
                        break
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}")
        
        return history
    
    def evaluate(self, df):
        """
        Evaluate the model
        
        Args:
            df (pandas.DataFrame): Data to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        # Preprocess data
        combined, combined_lengths, labels = self.preprocess_data(df)
        
        # Create dataset and dataloader
        dataset = TensorDataset(combined, combined_lengths, labels)
        # Use a fixed batch size for evaluation
        dataloader = DataLoader(dataset, batch_size=64)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Get batch data
                combined, combined_lengths, labels = batch
                
                # Sort by sequence length in descending order for pack_padded_sequence
                combined_lengths, sorted_idx = combined_lengths.sort(descending=True)
                combined = combined[sorted_idx]
                labels = labels[sorted_idx]
                
                # Move to device
                combined = combined.to(self.device)
                combined_lengths = combined_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(combined, combined_lengths)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=[self.label_map_inv[i] for i in range(3)],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, cm, accuracy=None, title='BiLSTM Attention Confusion Matrix', save_dir=None):
        """
        Plot confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            accuracy (float, optional): Model accuracy to include in the title
            title (str): Plot title
            save_dir (str): Directory to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.label_map.keys()),
                   yticklabels=list(self.label_map.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Include accuracy in the title if provided
        if accuracy is not None:
            plt.title(f'{title} (Accuracy: {accuracy:.4f})')
        else:
            plt.title(title)
        
        # Save the figure
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.png"))
            plt.close()
        else:
            plt.show()
    
    def save(self, filepath):
        """
        Save the model
        
        Args:
            filepath (str): Path to save the model
        """
        # Extract model parameters for filename
        param_str = "_".join([f"{k}={v}" for k, v in self.config.items() if not isinstance(v, list) and k not in ['max_sequence_length', 'vocab_size', 'batch_size', 'validation_split', 'early_stopping_patience']])
        
        # Save model with descriptive filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer': self.tokenizer,
            'label_map': self.label_map,
            'label_map_inv': self.label_map_inv
        }, filepath)
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath)
        
        # Load config
        self.config = checkpoint['config']
        
        # Update model parameters
        self.vocab_size = self.config.get('vocab_size', 30000)
        self.embedding_dim = self.config.get('embedding_dim', 300)
        self.hidden_dim = self.config.get('lstm_units', 128)
        self.n_layers = self.config.get('n_layers', 1)
        self.dropout = self.config.get('dropout_rate', 0.3)
        
        # Recreate model
        self.model = BiLSTMAttention(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=3,
            n_layers=self.n_layers,
            bidirectional=True,
            dropout=self.dropout,
            pad_idx=0
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = checkpoint['tokenizer']
        
        # Load label maps
        self.label_map = checkpoint['label_map']
        self.label_map_inv = checkpoint['label_map_inv']
