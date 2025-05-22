# Assignment 1: Neural Networks & Machine Learning Pipelines

**Course Assignment for Gabriel, Spiking Neurons**

---

Welcome to Assignment 1! This repository contains a multi-part machine learning project focused on neural networks, deep learning, and best practices in ML engineering. The assignment is structured into three main parts, each building on the previous, and is designed to demonstrate both foundational understanding and practical engineering skills.

## 📁 Repository Structure

```
Assignment 1/
│
├── Part 1/
├── Part 2/
├── Part 3/
└── README.md (this file)
```

---

## 🧑‍🏫 Assignment Context
This project was created as part of a course assignment for **Gabriel** at **Spiking Neurons**. It covers the full pipeline from building neural networks from scratch to applying advanced deep learning techniques on real-world data, with a strong focus on reproducibility, reporting, and MLOps best practices.

---

## 📦 Parts Overview

### Part 1: Neural Networks from Scratch
- **Goal:** Build up from a single neuron (manual Python, then NumPy) to a full artificial neural network (ANN) using PyTorch, and finally run on CUDA GPU.
- **Contents:**
  - `neuron_manual.py`: Single neuron, manual implementation
  - `neuron_numpy.py`: Single neuron, NumPy implementation
  - `ann_numpy_layer.py`: ANN layer with NumPy
  - `ann_pytorch_model.py`: ANN in PyTorch (CPU)
  - `ann_pytorch_cuda.py`: ANN in PyTorch (GPU/CUDA)
  - `mnist_loader.py`: MNIST data loader
  - `requirements.txt`: Dependencies
  - `README.md` & `reflection.txt`: Documentation and learning reflections
- **Note:** Data is downloaded automatically by the scripts; no need to include raw data.

---

### Part 2: ML Pipeline & MLOps for MNIST (CNN)
- **Goal:** Build a robust, modular ML pipeline for MNIST digit classification using PyTorch and CNNs, with MLOps best practices (logging, checkpointing, experiment tracking, data augmentation, regularization, reproducibility).
- **Contents:**
  - `scripts/`: Modular scripts for training, model definition, data augmentation, utilities, and hyperparameter search
  - `logs/`, `checkpoints/`: Output folders (can be regenerated)
  - `requirements.txt`: Dependencies
  - `README.md` & `reflection.txt`: Documentation and learning reflections
- **Note:** All outputs are generated per run; only code and documentation are needed for review.

---

### Part 3: Brain Tumor MRI Classification (Classic & Transfer Learning)
- **Goal:** Apply both a custom CNN and transfer learning (ResNet50) to classify brain tumor MRI images (4 classes), with strong data augmentation, class weighting, and best practices for reporting and reproducibility.
- **Contents:**
  - `train_classic.py`: Custom CNN pipeline
  - `train_transfer.py`: Transfer learning pipeline (ResNet50)
  - `Training/`, `Testing/`: Data folders (not included in submission; see README for download instructions)
  - `logs/`: Output folders (can be regenerated)
  - `requirements.txt`: Dependencies
  - `README.md` & `reflection.txt`: Documentation and learning reflections
- **Note:** Dataset is not included; instructions provided for obtaining data.

---

## 🚀 How to Use
1. **Clone the repository**
2. **Install dependencies** for each part using the provided `requirements.txt`
3. **Follow the README in each part** for detailed instructions on running experiments, training models, and reproducing results

---

## 📝 Reporting & Reproducibility
- Each part includes a `README.md` and `reflection.txt` with detailed documentation, experiment logs, and reflections on the learning process.
- All code follows best practices for reproducibility and reporting, including output structure, logging, and experiment tracking.

---

## 🙏 Acknowledgements
Assignment created for **Gabriel** at **Spiking Neurons**. Special thanks for guidance and feedback throughout the project.

---

For any questions, please refer to the part-specific README files or contact the course instructor.
