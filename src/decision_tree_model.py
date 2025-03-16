"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This module implements a Decision Tree model for NLI.
It's a simple and interpretable approach that doesn't use CNN or RoBERTa.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

class DecisionTreeModel:
    """Decision Tree model for NLI"""
    
    def __init__(self, config):
        """
        Initialize the Decision Tree model
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.max_features = config.get('max_features', 5000)
        self.ngram_range_min = config.get('ngram_range_min', 1)
        self.ngram_range_max = config.get('ngram_range_max', 2)
        self.ngram_range = (self.ngram_range_min, self.ngram_range_max)
        self.max_depth = config.get('max_depth', 5)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.criterion = config.get('criterion', 'gini')
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.label_map_inv = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        
        # Create pipelines for text and assertion
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            ))
        ])
        
        self.assertion_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            ))
        ])
        
        # Create a pipeline for combined features
        self.combined_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            ))
        ])
    
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
        print("Training Decision Tree model...")
        
        # Convert labels to indices
        y_train = train_df['label'].map(self.label_map).values
        
        # Train text pipeline
        print("Training text pipeline...")
        self.text_pipeline.fit(train_df['text'], y_train)
        
        # Train assertion pipeline
        print("Training assertion pipeline...")
        self.assertion_pipeline.fit(train_df['assertion'], y_train)
        
        # Train combined pipeline
        print("Training combined pipeline...")
        combined_text = train_df['text'] + " [SEP] " + train_df['assertion']
        self.combined_pipeline.fit(combined_text, y_train)
        
        # Return empty history (not applicable for Decision Tree)
        return {}
    
    def predict(self, df):
        """
        Make predictions
        
        Args:
            df (pandas.DataFrame): Data to predict
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        # Get predictions from each pipeline
        text_preds = self.text_pipeline.predict_proba(df['text'])
        assertion_preds = self.assertion_pipeline.predict_proba(df['assertion'])
        
        # Get predictions from combined pipeline
        combined_text = df['text'] + " [SEP] " + df['assertion']
        combined_preds = self.combined_pipeline.predict_proba(combined_text)
        
        # Ensemble predictions (average probabilities)
        ensemble_preds = (text_preds + assertion_preds + combined_preds) / 3
        
        # Get final predictions
        return np.argmax(ensemble_preds, axis=1)
    
    def evaluate(self, df):
        """
        Evaluate the model
        
        Args:
            df (pandas.DataFrame): Data to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        # Convert labels to indices
        y_true = df['label'].map(self.label_map).values
        
        # Make predictions
        y_pred = self.predict(df)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=[self.label_map_inv[i] for i in range(3)],
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, cm, title='Decision Tree Confusion Matrix', save_dir=None):
        """
        Plot confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            title (str): Plot title
            save_dir (str): Directory to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.label_map.keys()),
                   yticklabels=list(self.label_map.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # Save the figure
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}.png"))
        else:
            os.makedirs('results', exist_ok=True)
            plt.savefig(f"results/{title.lower().replace(' ', '_')}.png")
        plt.close()
    
    def save(self, base_path):
        """
        Save all models and results
        
        Args:
            base_path (str): Base directory path to save models and results
        """
        # Extract model parameters for filename
        param_str = f"max_depth={self.max_depth}_min_samples_split={self.min_samples_split}_max_features={self.max_features}"
        
        # Create directory with parameters in name
        save_dir = os.path.dirname(base_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract filename from base_path
        filename = os.path.basename(base_path)
        
        # Save individual models with descriptive names
        joblib.dump(self.text_pipeline, os.path.join(save_dir, f'text_{filename}'))
        joblib.dump(self.assertion_pipeline, os.path.join(save_dir, f'assertion_{filename}'))
        joblib.dump(self.combined_pipeline, os.path.join(save_dir, f'combined_{filename}'))
        
        # Save metadata
        joblib.dump({
            'config': self.config,
            'label_map': self.label_map,
            'label_map_inv': self.label_map_inv
        }, os.path.join(save_dir, 'metadata.pkl'))
        
        return save_dir
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath (str): Path to load the model from
        """
        data = joblib.load(filepath)
        self.text_pipeline = data['text_pipeline']
        self.assertion_pipeline = data['assertion_pipeline']
        self.combined_pipeline = data['combined_pipeline']
        self.config = data['config']
        self.label_map = data['label_map']
        self.label_map_inv = data['label_map_inv']
