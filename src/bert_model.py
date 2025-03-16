import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

class BERTModel:
    """BERT model for NLI using Hugging Face transformers"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3).to(self.device)
        
        # Training parameters
        learning_rate = float(config.get('learning_rate', '2e-5'))
        self.optimizer = torch.optim.AdamW(list(self.bert.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.label_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        
    def preprocess_data(self, df):
        # Tokenize text and assertion pairs
        texts = df['text'].tolist()
        assertions = df['assertion'].tolist()
        labels = df['label'].map(self.label_map).values
        
        # Tokenize with BERT tokenizer
        encodings = self.tokenizer(
            texts,
            assertions,
            padding=True,
            truncation=True,
            max_length=self.config.get('max_sequence_length', 64),
            return_tensors='pt'
        )
        
        return encodings, torch.tensor(labels)
    
    def train_model(self, train_df, dev_df=None, epochs=2, batch_size=16, validation_split=None):
        train_encodings, train_labels = self.preprocess_data(train_df)
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training loop
        self.bert.train()
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :]
                predictions = self.classifier(logits)
                
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
            
            # Evaluate on dev set if provided
            if dev_df is not None:
                self.evaluate(dev_df)
        
        return {'loss': total_loss/len(train_loader)}
    
    def evaluate(self, df):
        encodings, labels = self.preprocess_data(df)
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        self.bert.eval()
        self.classifier.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :]
                predictions = self.classifier(logits)
                
                all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=list(self.label_map.keys()),
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def save(self, model_dir):
        """Save model components to disk"""
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.bert.state_dict(), os.path.join(model_dir, 'bert_model.pt'))
        torch.save(self.classifier.state_dict(), os.path.join(model_dir, 'classifier.pt'))
        joblib.dump(self.config, os.path.join(model_dir, 'config.pkl'))
        self.tokenizer.save_pretrained(model_dir)
        
        return model_dir
    
    def load(self, model_dir):
        """Load model components from disk"""
        self.bert.load_state_dict(torch.load(os.path.join(model_dir, 'bert_model.pt')))
        self.classifier.load_state_dict(torch.load(os.path.join(model_dir, 'classifier.pt')))
        self.config = joblib.load(os.path.join(model_dir, 'config.pkl'))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
