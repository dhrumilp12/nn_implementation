# nn_implementation

A Python-only repository implementing and analyzing a neural network on the Fashion-MNIST dataset. It includes reusable modules, an experiment runner, a saved model checkpoint, and figures that visualize training and evaluation results.

Languages: Python (100%)

## Table of Contents
- Overview
- Repository layout
- Setup
- Usage
- Results and artifacts
- Development notes
- License

## Overview

This project provides a simple, modular neural network implementation and training pipeline, likely using PyTorch (inferred from the `.pt` checkpoint). Code is organized into:
- An experiment runner to orchestrate training/evaluation
- Custom layers/modules and utility functions
- Saved model weights and generated analysis plots

## Repository layout

Top-level:
- [.gitignore](https://github.com/dhrumilp12/nn_implementation/blob/main/.gitignore) — Common Python ignores.
- [.gitattributes](https://github.com/dhrumilp12/nn_implementation/blob/main/.gitattributes) — Attributes for Git.
- [.DS_Store](https://github.com/dhrumilp12/nn_implementation/blob/main/.DS_Store) — macOS metadata (can be ignored).

Package directory: [nn_implementation_code/](https://github.com/dhrumilp12/nn_implementation/tree/main/nn_implementation_code)
- [__init__.py](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/__init__.py) — Marks the directory as a Python package.
- [base_experiment.py](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/base_experiment.py) — Experiment/training loop orchestration.
- [custom_functions.py](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/custom_functions.py) — Utility functions (e.g., metrics, data handling, helpers).
- [custom_modules.py](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/custom_modules.py) — Model components or layers.
- [data/](https://github.com/dhrumilp12/nn_implementation/tree/main/nn_implementation_code/data) — Placeholder for data assets (empty in repo).
- [fashion_mnist_sgd.pt](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/fashion_mnist_sgd.pt) — Saved model checkpoint (likely trained with SGD).
- [q5_first_misclassified_per_class.png](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/q5_first_misclassified_per_class.png) — Analysis figure (example of misclassifications).
- [q6_test_loss_vs_epoch.png](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/q6_test_loss_vs_epoch.png) — Test loss curve.
- [q6_train_loss_vs_epoch.png](https://github.com/dhrumilp12/nn_implementation/blob/main/nn_implementation_code/q6_train_loss_vs_epoch.png) — Train loss curve.

## Setup

Requirements (typical for PyTorch projects):
- Python 3.9+ recommended
- PyTorch and torchvision
- matplotlib, numpy

Example:
```sh
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install torch torchvision matplotlib numpy
```

## Usage

As the repository structure implies modular code, a typical usage pattern is:
- Edit or create a small script to import `base_experiment.py`, define hyperparameters, and run training/evaluation.
- Alternatively, if `base_experiment.py` can be run directly, invoke it with appropriate arguments. Check the file’s CLI (if present) or adapt the import example below.

Example: running an experiment from a driver script
```python
# run_experiment.py
from nn_implementation_code.base_experiment import main  # or a suitable entry function

if __name__ == "__main__":
    # Provide config via arguments or dict (adjust to match base_experiment.py interface)
    # e.g., epochs, batch_size, lr, model config, dataset path, etc.
    main(
        epochs=10,
        batch_size=64,
        learning_rate=0.01,
        optimizer="sgd",
        seed=42,
    )
```

Then:
```sh
python run_experiment.py
```

Loading the saved checkpoint:
```python
import torch
from nn_implementation_code.custom_modules import build_model  # hypothetical factory

model = build_model()
state = torch.load("nn_implementation_code/fashion_mnist_sgd.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

Note: Adjust imports and function names to match the actual APIs in `base_experiment.py` and `custom_modules.py`.

## Results and artifacts

The repository includes example outputs:
- Training/validation curves:
  - `q6_train_loss_vs_epoch.png`
  - `q6_test_loss_vs_epoch.png`
- Misclassification analysis:
  - `q5_first_misclassified_per_class.png`
- Trained weights:
  - `fashion_mnist_sgd.pt`

Use these to verify your local runs or to perform further analysis.

## Development notes

- `base_experiment.py` likely contains the training loop, evaluation, logging, and plotting hooks. Extend it to add features like:
  - Better CLI/argparse configuration
  - Checkpointing and resume
  - TensorBoard logging
- `custom_modules.py` and `custom_functions.py` are the main extension points for models and utilities:
  - Add architectures or preprocessing pipelines
  - Implement additional metrics or data transforms
- Place dataset caches or auxiliary files under `nn_implementation_code/data/` (keep large files out of Git; prefer downloads or caching at runtime).

## License

No explicit license file is present. If you plan to share or extend this work, consider adding a LICENSE (e.g., MIT, Apache-2.0, GPL-3.0) to clarify usage terms.

---
For questions or contributions, please open an issue or submit a PR.
