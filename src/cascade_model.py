"""
@Authors: Ilyes DJERFAF, Nazim KESKES

This module implements a Cascade model for NLI.
It uses a decision model (Md) to determine if a sample is entailment or contradiction,
then uses a positive model (Mp) or negative model (Mn) for the final prediction.
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

class CascadeModel:
    """Cascade model for NLI"""
    
    def __init__(self, config):
        """
        Initialize the Cascade model
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.max_features = config.get('tfidf_max_features', 5000)
        self.max_depth = config.get('max_depth', 10)
        self.min_samples_split = config.get('min_samples_split', 2)
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        self.label_map_inv = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
        
        # Create decision model (Md)
        self.md_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('classifier', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            ))
        ])
        
        # Create positive model (Mp)
        self.mp_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('classifier', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            ))
        ])
        
        # Create negative model (Mn)
        self.mn_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features)),
            ('classifier', DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            ))
        ])
    
    def train_model(self, train_df, dev_df=None, epochs=None, batch_size=None, validation_split=None):
        """
        Train the model
        
        Args:
            train_df (pandas.DataFrame): Training data
            dev_df (pandas.DataFrame, optional): Development data
            epochs (int, optional): Not used for Cascade model
            batch_size (int, optional): Not used for Cascade model
            validation_split (float, optional): Not used for Cascade model
            
        Returns:
            dict: Training history (empty for Cascade model)
        """
        print("Training Cascade model...")
        
        # Preprocess data
        train_df = self._preprocess_data(train_df)
        
        # Train decision model (Md)
        print("Training decision model (Md)...")
        # Filter data to include only 'entailment' and 'contradiction' labels
        filtered_data = train_df[train_df['label'].isin(['entailment', 'contradiction'])]
        
        # Split the data into features and target
        X_md = filtered_data['combined_text']
        y_md = filtered_data['label']
        
        # Train the model
        self.md_model.fit(X_md, y_md)
        print("Decision model (Md) training completed!")
        
        # Train positive model (Mp)
        print("Training positive model (Mp)...")
        
        # Get 50% entailment samples
        entailment_samples = train_df[train_df['label'] == 'entailment']
        entailment_count = len(entailment_samples)
        
        # Calculate how many other samples we need (equal to entailment_count)
        # 40% neutral and 10% contradiction
        neutral_count = int(0.8 * entailment_count)  # 40% of total = 80% of the other half
        contradiction_count = int(0.2 * entailment_count)  # 10% of total = 20% of the other half
        
        # Sample the required number of neutral and contradiction samples
        neutral_samples = train_df[train_df['label'] == 'neutral'].sample(n=neutral_count, random_state=42)
        contradiction_samples = train_df[train_df['label'] == 'contradiction'].sample(n=contradiction_count, random_state=42)
        
        # Combine the samples
        other_samples = pd.concat([neutral_samples, contradiction_samples])
        
        # Create a new dataframe with balanced classes
        balanced_data = pd.concat([entailment_samples, other_samples])
        
        # Create binary labels: 'entailment' remains 'entailment', others become 'neutral'
        balanced_data.loc[balanced_data['label'] != 'entailment', 'label'] = 'neutral'
        
        # Split the data into features and target
        X_mp = balanced_data['combined_text']
        y_mp = balanced_data['label']
        
        # Train the model
        self.mp_model.fit(X_mp, y_mp)
        print("Positive model (Mp) training completed!")
        
        # Train negative model (Mn)
        print("Training negative model (Mn)...")
        
        # Get 50% contradiction samples
        contradiction_samples = train_df[train_df['label'] == 'contradiction']
        contradiction_count = len(contradiction_samples)
        
        # Calculate how many other samples we need (equal to contradiction_count)
        # 40% neutral and 10% entailment
        neutral_count = int(0.8 * contradiction_count)  # 40% of total = 80% of the other half
        entailment_count = int(0.2 * contradiction_count)  # 10% of total = 20% of the other half
        
        # Sample the required number of neutral and entailment samples
        neutral_samples = train_df[train_df['label'] == 'neutral'].sample(n=neutral_count, random_state=42)
        entailment_samples = train_df[train_df['label'] == 'entailment'].sample(n=entailment_count, random_state=42)
        
        # Combine the samples
        other_samples = pd.concat([neutral_samples, entailment_samples])
        
        # Create a new dataframe with balanced classes
        balanced_data = pd.concat([contradiction_samples, other_samples])
        
        # Create binary labels: 'contradiction' remains 'contradiction', others become 'neutral'
        balanced_data.loc[balanced_data['label'] != 'contradiction', 'label'] = 'neutral'
        
        # Split the data into features and target
        X_mn = balanced_data['combined_text']
        y_mn = balanced_data['label']
        
        # Train the model
        self.mn_model.fit(X_mn, y_mn)
        print("Negative model (Mn) training completed!")
        
        # Return empty history (not applicable for Cascade model)
        return {}
    
    def _preprocess_data(self, data):
        """
        Preprocess the data
        
        Args:
            data (pandas.DataFrame): Data to preprocess
            
        Returns:
            pandas.DataFrame: Preprocessed data
        """
        # Create a new feature by combining text and assertion
        data['combined_text'] = data['text'] + ' ' + data['assertion']
        # Handle missing values
        data['combined_text'] = data['combined_text'].fillna('')
        return data
    
    def predict(self, df):
        """
        Make predictions using the cascade model
        
        Args:
            df (pandas.DataFrame): Data to predict
            
        Returns:
            numpy.ndarray: Predicted labels as indices
        """
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Get the combined text
        X = df['combined_text']
        
        # Initialize predictions array
        y_pred = np.zeros(len(df), dtype=int)
        
        # Make predictions for each sample
        for i, text in enumerate(X):
            # First, predict using the decision model (Md)
            md_prediction = self.md_model.predict([text])[0]
            
            # Based on the decision model's prediction, use either Mp or Mn
            if md_prediction == 'entailment':
                # Use the positive model (Mp)
                mp_prediction = self.mp_model.predict([text])[0]
                
                # Map the prediction to an index
                if mp_prediction == 'entailment':
                    y_pred[i] = self.label_map['entailment']
                else:  # mp_prediction == 'neutral'
                    y_pred[i] = self.label_map['neutral']
            else:  # md_prediction == 'contradiction'
                # Use the negative model (Mn)
                mn_prediction = self.mn_model.predict([text])[0]
                
                # Map the prediction to an index
                if mn_prediction == 'contradiction':
                    y_pred[i] = self.label_map['contradiction']
                else:  # mn_prediction == 'neutral'
                    y_pred[i] = self.label_map['neutral']
        
        return y_pred
    
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
    
    def plot_confusion_matrix(self, cm, accuracy=None, title='Cascade Confusion Matrix', save_dir=None):
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
        param_str = f"max_depth={self.max_depth}_min_samples_split={self.min_samples_split}_tfidf_max_features={self.max_features}"
        
        # Create directory with parameters in name
        save_dir = os.path.dirname(base_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract filename from base_path
        filename = os.path.basename(base_path)
        
        # Save the models
        model_data = {
            'md_model': self.md_model,
            'mp_model': self.mp_model,
            'mn_model': self.mn_model,
            'config': self.config,
            'label_map': self.label_map,
            'label_map_inv': self.label_map_inv
        }
        
        joblib.dump(model_data, base_path)
        
        return save_dir
    
    def load(self, filepath):
        """
        Load the model
        
        Args:
            filepath (str): Path to load the model from
        """
        data = joblib.load(filepath)
        self.md_model = data['md_model']
        self.mp_model = data['mp_model']
        self.mn_model = data['mn_model']
        self.config = data['config']
        self.label_map = data['label_map']
        self.label_map_inv = data['label_map_inv']
