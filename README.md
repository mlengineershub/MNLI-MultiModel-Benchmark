# Natural Language Inference (NLI) Approach with PyTorch

This project implements various models for Natural Language Inference (NLI) using the MultiNLI dataset with PyTorch. The models are trained to classify pairs of sentences as entailment, contradiction, or neutral.

## Project Structure

```
.
├── config/
│   └── configuration.yaml    # Configuration file for models
├── data/
│   ├── dev1.csv              # Development set 1 (for hyperparameter tuning)
│   ├── dev2.csv              # Development set 2 (for final evaluation)
│   ├── test.csv              # Test set
│   └── train.csv             # Training set
├── images/
│   ├── bi-lstm.png           # Bi-LSTM architecture diagram
│   ├── data-preparation.png  # Data preparation workflow
│   ├── model-training.png    # Model training process
│   ├── models.png            # Models architecture diagram
│   ├── parameters.png        # Hyperparameter configuration
│   └── pipeline.png          # Complete pipeline diagram
├── models/                   # Saved model weights
│   ├── bilstm_attention/     # BiLSTM with attention models
│   ├── cascade/              # Cascade models with different parameters
│   └── decision_tree/        # Decision tree models with different parameters
├── results/                  # Evaluation results
│   ├── benchmark/            # Model benchmark results and comparison
│   ├── bilstm_attention/     # BiLSTM results
│   ├── cascade/              # Cascade model results
│   └── decision_tree/        # Decision tree results and confusion matrices
├── src/
│   ├── __init__.py
│   ├── benchmark_models.py   # Script for benchmarking models
│   ├── bert_model.py         # BERT model implementation
│   ├── bilstm_attention_model.py # BiLSTM with attention implementation
│   ├── cascade.py            # Script for training and evaluating the cascade model
│   ├── cascade_model.py      # Cascade model implementation
│   ├── create_dummy_models.py # Utility for creating test models
│   ├── data_split.py         # Script for splitting data
│   ├── decision_tree_model.py # Decision tree model implementation
│   ├── find_and_benchmark.py # Script to find and benchmark models
│   ├── main.py               # Main script for running the code
│   ├── models.py             # Model factory and base classes
│   ├── preprocessing.py      # Data preprocessing utilities
│   ├── regenerate_confusion_matrices.py # Script to regenerate confusion matrices
│   ├── run_benchmark.py      # Wrapper for running benchmarks
│   └── train_evaluate.py     # Training and evaluation code
├── CONTRIBUTE.md             # Contribution guidelines
├── LICENSE                   # License information
├── README.md                 # This file
├── requirements.txt          # Required packages
└── setup.py                  # Package setup script
```

## Models

The following models are implemented:

1. **TF-IDF with Decision Tree**: A simple and interpretable model that uses TF-IDF features with a decision tree classifier. Multiple configurations are available with different hyperparameters:
   - Max depth: 10, 20
   - Min samples split: 2, 10
   - Max features: 0.3, 0.5
   - TF-IDF max features: 5000, 10000

2. **Bi-LSTM with Attention**: A bidirectional LSTM model with attention mechanism for better capturing the relationships between premise and hypothesis sentences.

3. **Cascade Model**: A cascade approach that uses three specialized models:
   - A decision model (Md) to determine if a sample is entailment or contradiction
   - A positive model (Mp) for entailment vs. neutral classification
   - A negative model (Mn) for contradiction vs. neutral classification
   
   This approach allows each model to focus on a specific aspect of the classification task, leading to improved overall performance.

4. **BERT Model**: A BERT-based model that leverages pre-trained contextual embeddings for NLI tasks.

## Pipeline

The project follows a comprehensive pipeline for NLI:

1. **Data Preparation**: Loading and preprocessing the MultiNLI dataset
2. **Model Training**: Training models with hyperparameter search
3. **Evaluation**: Evaluating models on development and test sets
4. **Results Analysis**: Analyzing confusion matrices and performance metrics

## Configuration

The `config/configuration.yaml` file contains hyperparameters for each model. The hyperparameters are specified as lists, and the training code will perform a grid search over all combinations of hyperparameters. This allows for extensive experimentation with different model configurations.

Example configuration:
```yaml
common:
  batch_size: 64
  validation_split: 0.1

decision_tree:
  max_depth: [10, 20]
  min_samples_split: [2, 10]
  max_features: [0.3, 0.5]
  tfidf_max_features: [5000, 10000]

bilstm_attention:
  embedding_dim: [100, 200]
  hidden_dim: [128, 256]
  attention_dim: [64, 128]
  dropout: [0.2, 0.5]
  epochs: 10
```

## Data

The data is split into training, development, and test sets. The development set is further split into dev1 and dev2. The training set is used to train the models, dev1 is used for hyperparameter tuning, and dev2 and test are used for final evaluation.

Each dataset contains the following columns:
- `label`: The label for the pair of sentences (neutral, entailment, or contradiction)
- `text`: The premise sentence
- `assertion`: The hypothesis sentence

## Model Benchmarking

The project includes a benchmarking system to compare different models on the test dataset. The benchmark system:

1. Loads pre-trained models from pickle files
2. Evaluates them on the test dataset
3. Generates confusion matrices
4. Calculates performance metrics
5. Saves the results to the `results/benchmark/` directory

### Running the Benchmark

To benchmark specific models:

```bash
python src/run_benchmark.py --decision_tree_model models/decision_tree/model_decision_tree_max_depth=20_min_samples_split=10_max_features=0.5_tfidf_max_features=10000.pkl --bilstm_model models/bilstm_attention/model_bilstm_attention_epochs=5_embedding_dim=200_lstm_units=64_learning_rate=0.001_n_layers=2.pkl --cascade_model models/cascade/model_cascade_max_depth=10_min_samples_split=2_tfidf_max_features=5000.pkl
```

To automatically find and benchmark the most recent models:

```bash
python src/find_and_benchmark.py
```

### Benchmark Results

The benchmark results are saved in the `results/benchmark/` directory:
- `decision_tree_confusion_matrix.png`: Confusion matrix for the Decision Tree model
- `bilstm_attention_confusion_matrix.png`: Confusion matrix for the BiLSTM model
- `cascade_confusion_matrix.png`: Confusion matrix for the Cascade model
- `benchmark_results.csv`: CSV file with accuracy results
- `README.md`: Documentation of the benchmark system and results

## Usage

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

### Training and Evaluation

To train and evaluate all models:

```bash
python src/main.py --train
```

To specify which models to train:

```bash
python src/main.py --train --models bilstm_attention decision_tree cascade
```

To specify a different configuration file:

```bash
python src/main.py --train --config path/to/config.yaml
```

To specify different data files:

```bash
python src/main.py --train --train_data path/to/train.csv --dev_data path/to/dev.csv
```

To train cascade pipeline:

```bash
python src/cascade.py
```

### Regenerating Confusion Matrices

If you need to regenerate confusion matrices for existing models:

```bash
python src/regenerate_confusion_matrices.py
```

## Results

The results of the training and evaluation are saved in the following directories:
- `results/`: Contains confusion matrix plots and evaluation metrics
  - Each model type has its own subdirectory (e.g., `results/decision_tree/`)
  - `all_results.csv` contains the performance metrics for all hyperparameter combinations
  - Confusion matrix images are saved with descriptive filenames indicating the hyperparameters used
- `models/`: Contains the best model weights
  - Models are saved as .pkl files with descriptive filenames

## Evaluation Metrics

The models are evaluated using the following metrics:
- Accuracy
- Precision, recall, and F1-score for each class (neutral, entailment, contradiction)
- Confusion matrix visualization

## Contributing

Please see the [CONTRIBUTE.md](CONTRIBUTE.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Authors

- Ilyes DJERFAF
- Nazim KESKES
