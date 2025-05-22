# Brain Tumor Classification (MRI) - Part 3

This project implements two approaches for classifying brain MRI images into four categories (glioma, meningioma, pituitary tumor, no tumor):

Link to GitHub repo: https://github.com/JSON-Bjorn/Assignment-1

Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

1. **Classic Learning:** Training a custom CNN from scratch.
2. **Transfer Learning:** Fine-tuning a pretrained ResNet50 model.

## Folder Structure

- `Training/` and `Testing/`: Contain images organized by class for training/validation and testing.
- `train_classic.py`: Train a simple CNN from scratch.
- `train_transfer.py`: Fine-tune a pretrained ResNet50.
- `logs/`: Contains a subfolder for each run (named by timestamp) with all outputs.
- `requirements.txt`: Dependencies.
- `reflection.txt`: Reflection and summary of the project.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train a classic CNN:**
   ```bash
   python train_classic.py
   ```
3. **Train with transfer learning (ResNet50):**
   ```bash
   python train_transfer.py
   ```

All outputs (training/validation loss and accuracy plots, confusion matrix, correct/incorrect examples, checkpoints, config, script copy) are saved in a new subfolder under `logs/` for each run.

## Notes
- Images are resized to 224x224.
- Training runs for 50 epochs.
- Data augmentation is applied to the training set.
- The best model (by validation accuracy) is used for test evaluation.
- See `reflection.txt` for a summary and discussion of results.

--- 