# Natural Language Inference (NLI) Approach with PyTorch

This project implements various models for Natural Language Inference (NLI) using the MultiNLI dataset with PyTorch. The models are trained to classify pairs of sentences as entailment, contradiction, or neutral.

## Project Structure

```
.
├── config/
│   └── configuration.yaml    # Configuration file for models
├── data/
│   ├── dev1.csv              # Development set 1
│   ├── dev2.csv              # Development set 2
│   ├── test.csv              # Test set
│   └── train.csv             # Training set
├── images/
│   ├── bi-lstm.png           # Bi-LSTM architecture diagram
│   └── models.png            # Models diagram
├── src/
│   ├── __init__.py
│   ├── data_split.py         # Script for splitting data
│   ├── main.py               # Main script for running the code
│   ├── models.py             # Model implementations
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── train_evaluate.py     # Training and evaluation code
└── README.md                 # This file
```

## Models

The following models are implemented:

1. **TF-IDF Model**: A simple model that uses TF-IDF features and a neural network classifier.
2. **Bi-LSTM with Attention**: A bidirectional LSTM model with attention mechanism.
3. **Decision Tree Model**: A simple and interpretable model that uses TF-IDF features with a decision tree classifier.
4. **Transformer Model**: A transformer-based model (implementation in progress).
5. **BERT Model**: A BERT-based model (implementation in progress).

## Configuration

The `config/configuration.yaml` file contains hyperparameters for each model. The hyperparameters are specified as lists, and the training code will perform a grid search over all combinations of hyperparameters.

## Data

The data is split into training, development, and test sets. The development set is further split into dev1 and dev2. The training set is used to train the models, dev1 is used for hyperparameter tuning, and dev2 and test are used for final evaluation.

Each dataset contains the following columns:
- `label`: The label for the pair of sentences (neutral, entailment, or contradiction)
- `text`: The premise sentence
- `assertion`: The hypothesis sentence

## Usage

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Training and Evaluation

To train and evaluate all models:

```bash
python src/main.py --train
```

To specify which models to train:

```bash
python src/main.py --train --models tfidf bilstm_attention
```

To specify a different configuration file:

```bash
python src/main.py --train --config path/to/config.yaml
```

To specify different data files:

```bash
python src/main.py --train --train_data path/to/train.csv --dev_data path/to/dev.csv
```

### Results

The results of the training and evaluation are saved in the following directories:
- `results/`: Contains confusion matrix plots and evaluation metrics
- `models/`: Contains the best model weights (saved as .pkl files)

## Evaluation Metrics

The models are evaluated using the following metrics:
- Accuracy
- Precision, recall, and F1-score for each class
- Confusion matrix

## Authors

- Ilyes DJERFAF
- Nazim KESKES
