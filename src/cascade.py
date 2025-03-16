import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
def load_data():
    print("Loading data...")
    train_data = pd.read_csv('data/train.csv')
    print(f"Dataset shape: {train_data.shape}")
    return train_data

# Preprocess the data
def preprocess_data(data):
    print("Preprocessing data...")
    # Create a new feature by combining text and assertion
    data['combined_text'] = data['text'] + ' ' + data['assertion']
    return data

# Train the decision model (Md)
def train_decision_model(data):
    print("Training the decision model (Md)...")
    # Filter data to include only 'entailment' and 'contradiction' labels
    filtered_data = data[data['label'].isin(['entailment', 'contradiction'])]
    
    # Split the data into features and target
    X = filtered_data['combined_text']
    y = filtered_data['label']
    
    # Create a pipeline with TF-IDF vectorizer and Decision Tree classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    print("Decision model (Md) training completed!")
    return model

# Train the positive model (Mp)
def train_positive_model(data):
    print("Training the positive model (Mp)...")
    
    # Get 50% entailment samples
    entailment_samples = data[data['label'] == 'entailment']
    entailment_count = len(entailment_samples)
    
    # Calculate how many other samples we need (equal to entailment_count)
    # 40% neutral and 10% contradiction
    neutral_count = int(0.8 * entailment_count)  # 40% of total = 80% of the other half
    contradiction_count = int(0.2 * entailment_count)  # 10% of total = 20% of the other half
    
    # Sample the required number of neutral and contradiction samples
    neutral_samples = data[data['label'] == 'neutral'].sample(n=neutral_count, random_state=42)
    contradiction_samples = data[data['label'] == 'contradiction'].sample(n=contradiction_count, random_state=42)
    
    # Combine the samples
    other_samples = pd.concat([neutral_samples, contradiction_samples])
    
    # Create a new dataframe with balanced classes
    balanced_data = pd.concat([entailment_samples, other_samples])
    
    # Create binary labels: 'entailment' remains 'entailment', others become 'neutral'
    balanced_data.loc[balanced_data['label'] != 'entailment', 'label'] = 'neutral'
    
    # Split the data into features and target
    X = balanced_data['combined_text']
    y = balanced_data['label']
    
    # Create a pipeline with TF-IDF vectorizer and Decision Tree classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    print("Positive model (Mp) training completed!")
    return model

# Train the negative model (Mn)
def train_negative_model(data):
    print("Training the negative model (Mn)...")
    
    # Get 50% contradiction samples
    contradiction_samples = data[data['label'] == 'contradiction']
    contradiction_count = len(contradiction_samples)
    
    # Calculate how many other samples we need (equal to contradiction_count)
    # 40% neutral and 10% entailment
    neutral_count = int(0.8 * contradiction_count)  # 40% of total = 80% of the other half
    entailment_count = int(0.2 * contradiction_count)  # 10% of total = 20% of the other half
    
    # Sample the required number of neutral and entailment samples
    neutral_samples = data[data['label'] == 'neutral'].sample(n=neutral_count, random_state=42)
    entailment_samples = data[data['label'] == 'entailment'].sample(n=entailment_count, random_state=42)
    
    # Combine the samples
    other_samples = pd.concat([neutral_samples, entailment_samples])
    
    # Create a new dataframe with balanced classes
    balanced_data = pd.concat([contradiction_samples, other_samples])
    
    # Create binary labels: 'contradiction' remains 'contradiction', others become 'neutral'
    balanced_data.loc[balanced_data['label'] != 'contradiction', 'label'] = 'neutral'
    
    # Split the data into features and target
    X = balanced_data['combined_text']
    y = balanced_data['label']
    
    # Create a pipeline with TF-IDF vectorizer and Decision Tree classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    print("Negative model (Mn) training completed!")
    return model

# Predict using the cascade model
def predict_cascade(text, md_model, mp_model, mn_model):
    # First, predict using the decision model (Md)
    md_prediction = md_model.predict([text])[0]
    
    # Based on the decision model's prediction, use either Mp or Mn
    if md_prediction == 'entailment':
        # Use the positive model (Mp)
        final_prediction = mp_model.predict([text])[0]
    else:  # md_prediction == 'contradiction'
        # Use the negative model (Mn)
        final_prediction = mn_model.predict([text])[0]
    
    return final_prediction

# Evaluate the cascade model
def evaluate_cascade(data, md_model, mp_model, mn_model):
    print("Evaluating the cascade model...")
    # Get the combined text and true labels
    X = data['combined_text']
    y_true = data['label']
    
    # Make predictions using the cascade model
    y_pred = []
    for text in X:
        prediction = predict_cascade(text, md_model, mp_model, mn_model)
        y_pred.append(prediction)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report

# Main function
def main():
    # Load and preprocess the data
    data = load_data()
    data = preprocess_data(data)
    
    # Train the models
    md_model = train_decision_model(data)
    mp_model = train_positive_model(data)
    mn_model = train_negative_model(data)
    
    # Evaluate the cascade model
    evaluate_cascade(data, md_model, mp_model, mn_model)
    
    print("Cascade model implementation completed!")

if __name__ == "__main__":
    main()
