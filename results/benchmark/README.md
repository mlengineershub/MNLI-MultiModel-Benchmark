# Model Benchmark

This directory contains benchmark results comparing the Decision Tree and BiLSTM models on the test dataset.

## Scripts

The following scripts are available for benchmarking:

1. `src/benchmark_models.py` - Main script for benchmarking models
2. `src/run_benchmark.py` - Simple wrapper for running the benchmark
3. `src/find_and_benchmark.py` - Automatically finds model files and runs the benchmark
4. `src/create_dummy_models.py` - Creates dummy model files for testing

## Usage

### Option 1: Using find_and_benchmark.py (Recommended)

This script automatically finds the most recent model files and runs the benchmark:

```bash
python src/find_and_benchmark.py
```

If you want to specify the model files:

```bash
python src/find_and_benchmark.py --decision_tree_model models/decision_tree/your_dt_model.pkl --bilstm_model models/bilstm_attention/your_bilstm_model.pkl
```

### Option 2: Using run_benchmark.py

This script requires you to specify the model files:

```bash
python src/run_benchmark.py --decision_tree_model models/decision_tree/your_dt_model.pkl --bilstm_model models/bilstm_attention/your_bilstm_model.pkl
```

### Option 3: Using benchmark_models.py directly

```bash
python src/benchmark_models.py --decision_tree_model models/decision_tree/your_dt_model.pkl --bilstm_model models/bilstm_attention/your_bilstm_model.pkl --test_data data/test.csv --output_dir results/benchmark
```

### Creating Dummy Models for Testing

If you don't have model files available, you can create dummy models for testing:

```bash
python src/create_dummy_models.py
```

## Output

The benchmark generates the following outputs:

1. Confusion matrices for both models
2. A CSV file with accuracy results
3. Console output with model comparison

## Example Results

The benchmark compares the models on the test dataset and reports:

- Accuracy for each model
- Confusion matrices
- Classification reports
- Comparison of which model performed better
