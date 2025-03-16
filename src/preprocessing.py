"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This module contains utilities for preprocessing NLI data.
It includes:
- TF-IDF vectorization
- Text cleaning functions
- Feature extraction utilities
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Class for text preprocessing"""
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize the text preprocessor
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): n-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer_text = None
        self.tfidf_vectorizer_assertion = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and stopwords
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def fit_transform(self, df):
        """
        Fit TF-IDF vectorizers and transform data
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'assertion' columns
            
        Returns:
            tuple: TF-IDF features for text and assertion
        """
        # Clean text
        cleaned_text = df['text'].apply(self.clean_text)
        cleaned_assertion = df['assertion'].apply(self.clean_text)
        
        # Initialize TF-IDF vectorizers if not already initialized
        if self.tfidf_vectorizer_text is None:
            self.tfidf_vectorizer_text = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )
            self.tfidf_vectorizer_assertion = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )
            
            # Fit vectorizers
            text_features = self.tfidf_vectorizer_text.fit_transform(cleaned_text)
            assertion_features = self.tfidf_vectorizer_assertion.fit_transform(cleaned_assertion)
        else:
            # Transform using pre-fitted vectorizers
            text_features = self.tfidf_vectorizer_text.transform(cleaned_text)
            assertion_features = self.tfidf_vectorizer_assertion.transform(cleaned_assertion)
        
        return text_features, assertion_features
    
    def transform(self, df):
        """
        Transform data using pre-fitted TF-IDF vectorizers
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'assertion' columns
            
        Returns:
            tuple: TF-IDF features for text and assertion
        """
        # Check if vectorizers are fitted
        if self.tfidf_vectorizer_text is None or self.tfidf_vectorizer_assertion is None:
            raise ValueError("Vectorizers not fitted. Call fit_transform first.")
        
        # Clean text
        cleaned_text = df['text'].apply(self.clean_text)
        cleaned_assertion = df['assertion'].apply(self.clean_text)
        
        # Transform using pre-fitted vectorizers
        text_features = self.tfidf_vectorizer_text.transform(cleaned_text)
        assertion_features = self.tfidf_vectorizer_assertion.transform(cleaned_assertion)
        
        return text_features, assertion_features


class TfidfDataset(Dataset):
    """Dataset for TF-IDF features"""
    
    def __init__(self, features, labels):
        """
        Initialize the dataset
        
        Args:
            features (numpy.ndarray): TF-IDF features
            labels (numpy.ndarray): Labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }


class TfidfModel(nn.Module):
    """Model that uses TF-IDF features for NLI"""
    
    def __init__(self, config):
        """
        Initialize the TF-IDF model
        
        Args:
            config (dict): Configuration parameters
        """
        super().__init__()
        self.config = config
        self.max_features = config.get('max_features', 10000)
        
        # Ensure ngram_range is a tuple
        ngram_range = config.get('ngram_range', (1, 2))
        if isinstance(ngram_range, str):
            # Convert string representation to tuple
            ngram_range = tuple(map(int, ngram_range.strip('()').split(',')))
        self.ngram_range = ngram_range
        
        self.preprocessor = TextPreprocessor(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_built = False
    
    def build_model(self, input_dim):
        """
        Build the model architecture
        
        Args:
            input_dim (int): Input dimension (number of features)
        """
        # Define layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        
        self.model_built = True
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def preprocess_data(self, train_df, dev_df=None):
        """
        Preprocess data using TF-IDF
        
        Args:
            train_df (pandas.DataFrame): Training data
            dev_df (pandas.DataFrame, optional): Development data
            
        Returns:
            tuple: Preprocessed data
        """
        # Convert labels to indices
        label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        y_train = train_df['label'].map(label_map).values
        
        # Fit and transform training data
        X_train_text, X_train_assertion = self.preprocessor.fit_transform(train_df)
        
        # Combine text and assertion features
        X_train = np.hstack([X_train_text.toarray(), X_train_assertion.toarray()])
        
        if dev_df is not None:
            # Convert labels to indices
            y_dev = dev_df['label'].map(label_map).values
            
            # Transform development data
            X_dev_text, X_dev_assertion = self.preprocessor.transform(dev_df)
            
            # Combine text and assertion features
            X_dev = np.hstack([X_dev_text.toarray(), X_dev_assertion.toarray()])
            
            return (X_train, y_train), (X_dev, y_dev)
        
        return (X_train, y_train), None
    
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
        (X_train, y_train), val_data = self.preprocess_data(train_df, dev_df)
        
        # Create training dataset and dataloader
        train_dataset = TfidfDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataset and dataloader if provided
        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = TfidfDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        elif validation_split > 0:
            # Split training data for validation
            val_size = int(len(train_dataset) * validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Build model if not already built
        if not self.model_built:
            self.build_model(X_train.shape[1])
        
        # Move model to device
        self.to(self.device)
        
        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        
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
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Move batch to device
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(features)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * features.size(0)
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
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        # Move batch to device
                        features = batch['features'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        # Forward pass
                        outputs = self(features)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)
                        
                        # Update statistics
                        val_loss += loss.item() * features.size(0)
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
                    best_state_dict = self.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        # Restore best weights
                        self.load_state_dict(best_state_dict)
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
        # Convert labels to indices
        label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        y_true = df['label'].map(label_map).values
        
        # Transform data
        X_text, X_assertion = self.preprocessor.transform(df)
        X = np.hstack([X_text.toarray(), X_assertion.toarray()])
        
        # Create dataset and dataloader
        dataset = TfidfDataset(X, y_true)
        dataloader = DataLoader(dataset, batch_size=64)
        
        # Move model to device
        self.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Evaluation
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self(features)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item() * features.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        loss = total_loss / len(dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Get classification report
        label_map_inv = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        report = classification_report(
            all_labels, all_preds,
            target_names=[label_map_inv[i] for i in range(3)],
            output_dict=True
        )
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def save(self, filepath):
        """
        Save the model
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'preprocessor': self.preprocessor,
            'input_dim': self.fc1.in_features
        }, filepath)
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.config = checkpoint['config']
        self.preprocessor = checkpoint['preprocessor']
        
        # Build model with the correct input dimension
        self.build_model(checkpoint['input_dim'])
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
