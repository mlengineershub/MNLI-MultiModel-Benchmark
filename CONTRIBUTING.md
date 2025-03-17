# Contributing to MNLI-MultiModel-Benchmark

Thank you for your interest in contributing to the Multi Natural Language Inference (MNLI) Approach project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Adding New Features](#adding-new-features)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)
- [Testing](#testing)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/mlengineershub/NLI-Approach.git
   cd NLI-Approach
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/mlengineershub/NLI-Approach.git
   ```
4. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Make sure you can run the existing models:
   ```bash
   python src/main.py --train --models decision_tree
   ```

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Keep functions focused on a single responsibility
- Comment complex code sections
- Use type hints where appropriate

Example:
```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess the input text by converting to lowercase, removing punctuation,
    and optionally removing stopwords.
    
    Args:
        text: The input text to preprocess
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        The preprocessed text
    """
    # Implementation here
    return processed_text
```

## Adding New Features

### Adding a New Model

1. Create a new file in the `src/` directory for your model (e.g., `src/your_model.py`)
2. Implement your model following the existing patterns
3. Update `src/models.py` to include your model in the model factory
4. Add appropriate hyperparameters to `config/configuration.yaml`
5. Add tests for your model
6. Update documentation to reflect the new model

### Adding a New Dataset

1. Ensure the dataset follows the same format as existing datasets
2. Add preprocessing code if needed
3. Update documentation to include information about the new dataset

## Pull Request Process

1. Update your fork with the latest changes from the upstream repository:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Make sure your code passes all tests:
   ```bash
   # Add test command here
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request on GitHub with a clear title and description:
   - Describe what changes you've made
   - Reference any related issues
   - Explain how to test your changes
   - Include any necessary documentation updates

5. Address any feedback from code reviews

## Reporting Issues

When reporting issues, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, package versions)
6. Any relevant logs or error messages

## Documentation

- Update the README.md when adding new features or changing existing functionality
- Add docstrings to all new functions, classes, and modules
- Include examples where appropriate
- Update any diagrams or visualizations if necessary

## Testing

- Write tests for all new functionality
- Ensure existing tests pass with your changes
- Test your changes with different hyperparameter configurations
- Verify that your model works with the existing pipeline

## Model Evaluation

When adding a new model, please include:

1. Performance metrics on the development and test sets
2. Confusion matrices
3. Comparison with existing models
4. Analysis of strengths and weaknesses

Thank you for contributing to the NLI-Approach project!
