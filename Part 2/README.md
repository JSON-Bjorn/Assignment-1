# Assignment 1 - Part 2: Power-ups (Complete ML Pipeline)

This folder contains a **fully-featured, reproducible machine learning pipeline** for MNIST digit classification using PyTorch and Convolutional Neural Networks (CNNs). The pipeline is designed to follow best MLOps and ML reporting practices, including robust logging, checkpointing, plotting, and experiment versioning.

---

## Folder Structure

- `checkpoints/`  
  (Legacy) Used for storing model checkpoints in earlier versions. **Now, each run has its own output folder.**

- `logs/`  
  Contains a subfolder for each training run (named by timestamp, e.g. `run_20240501_153000`). Each run folder contains:
  - `checkpoints/` — Best model(s) and config for that run
  - `metrics.json` — All per-epoch metrics (loss, accuracy)
  - `epoch_times.json` — Per-epoch timing
  - `timing.json` — Total training time
  - `config.json` — Hyperparameters and run config
  - `loss_curve.png` — Plot of training/validation loss
  - `accuracy_curve.png` — Plot of training/validation accuracy
  - `confusion_matrix.png` — Confusion matrix for test set
  - `correct_examples.png` — Example images correctly classified
  - `incorrect_examples.png` — Example images misclassified
  - `train_copy.py` — Copy of the training script used for this run
  - `notebook_note.txt` — Reminder to save notebook if run in a notebook

- `scripts/`  
  All code for training, model definition, data augmentation, utilities, and experiment management:
    - `train.py` — Main training and evaluation script
    - `model.py` — CNN model definition
    - `data_augmentation.py` — Data augmentation transforms for MNIST
    - `utils.py` — Utility functions for checkpointing, logging, timing
    - `hyperparam_search.py` — (Placeholder) For future hyperparameter tuning

- `requirements.txt`  
  All dependencies for this pipeline (PyTorch, torchvision, numpy, matplotlib, seaborn, scikit-learn)

---

## What the Program Does

- **Trains a CNN on MNIST** with data augmentation, dropout, and batch normalization
- **Logs all hyperparameters and metrics** for each run
- **Saves a unique output folder** for every training run, containing all results and artifacts
- **Plots and saves**:
  - Training and validation loss curves
  - Training and validation accuracy curves
  - Confusion matrix for the test set
  - Example images of correct and incorrect predictions
- **Saves the best model checkpoint** (by validation accuracy) and its config
- **Logs per-epoch and total training time**
- **Saves a copy of the training script** (and a notebook note) for reproducibility
- **Sets random seeds** for reproducibility
- **Reloads and evaluates the best model** on the test set after training
- **Follows all best practices** for ML reporting and experiment tracking

---

## How to Use

1. **Install dependencies** (in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script:**
   ```bash
   python scripts/train.py
   ```
   - All outputs will be saved in a new subfolder under `logs/` (named by timestamp).

3. **Review results:**
   - Open the output folder for your run (e.g., `logs/run_20240501_153000/`).
   - Inspect plots, metrics, confusion matrix, and example images.
   - The best model checkpoint and config are in the `checkpoints/` subfolder.
   - The script copy and notebook note are included for reproducibility.

---

## Output Artifacts Explained

- **metrics.json** — Per-epoch loss and accuracy (train/val)
- **epoch_times.json** — Time taken for each epoch
- **timing.json** — Total training time
- **config.json** — All hyperparameters and run settings
- **loss_curve.png** — Training/validation loss per epoch
- **accuracy_curve.png** — Training/validation accuracy per epoch
- **confusion_matrix.png** — Normalized confusion matrix for test set
- **correct_examples.png** — Example images classified correctly
- **incorrect_examples.png** — Example images classified incorrectly
- **train_copy.py** — Copy of the script used for this run
- **notebook_note.txt** — Reminder to save notebook if run in a notebook

---

## Reproducibility & Best Practices

- All random seeds are set for both torch and numpy
- All code, configs, and results are versioned per run
- No data leakage: test set is never used for training or validation
- All reporting (plots, confusion matrix, predictions) is automated
- Each run is fully self-contained and reproducible

---

## Extending the Pipeline

- Add new models or architectures in `model.py`
- Add new data augmentations in `data_augmentation.py`
- Implement hyperparameter search in `hyperparam_search.py`
- Integrate with experiment tracking tools (e.g., TensorBoard, MLflow, Weights & Biases)
- Use the output folders for easy comparison and analysis of different runs

---

## Requirements

See `requirements.txt` for all dependencies. Install with:
```bash
pip install -r requirements.txt
```

See reflection.txt for a summary of my thoughts.
---