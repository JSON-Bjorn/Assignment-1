# Artificial Neural Network Implementation (MNIST)

This is my first project in machine learning in my path to becoming an AI Engineer. This assignment is ment to start at the base level of implementing a single neuron network and proceeding from a single neuron to multiple.

This project eventually implements many different stages of an Artificial Neural Network (ANN), from a single neuron to a complete PyTorch model trained on the MNIST dataset.

## Project Structure

```
Part 1/
├── neuron_manual.py      # Part A1: Single neuron implementation (manual Python)
├── neuron_numpy.py       # Part A2: Single neuron implementation (NumPy)
├── ann_numpy_layer.py    # Part B: ANN layer implementation (NumPy)
├── mnist_loader.py       # Helper: Loads MNIST dataset using PyTorch
├── ann_pytorch_model.py  # Part C: Full ANN model and training (PyTorch, CPU/GPU)
├── ann_pytorch_cuda.py   # Part D: Full ANN model and training (PyTorch, explicitly CUDA)
├── requirements.txt      # Required Python libraries
└── README.md             # This file
```

## Parts of the Assignment

### Part A: Single Neuron

- **Purpose**: Understand the fundamental building block of an ANN.
- **A1 (`neuron_manual.py`)**: Implements a single neuron using basic Python math operations (loops, standard library). Includes `forward` pass and optional activation functions (Sigmoid, ReLU).
- **A2 (`neuron_numpy.py`)**: Re-implements the neuron using NumPy for efficient vectorized calculations (dot product). Demonstrates the performance benefits of NumPy.

### Part B: ANN Layer with NumPy

- **Purpose**: Scale up from a single neuron to a full layer using matrix operations.
- **`ann_numpy_layer.py`**: Implements a layer of neurons. Takes a batch of inputs (NumPy array) and computes the outputs for all neurons in the layer using matrix multiplication (`np.dot`) between inputs and the weight matrix, plus a bias vector. Includes activation functions. This part focuses only on the _forward pass_.

### Part C: ANN Model in PyTorch

- **Purpose**: Build and train a complete ANN using the PyTorch deep learning framework.
- **`mnist_loader.py`**: Contains a function `get_mnist_loaders` to download, transform (normalize, convert to tensor), and create `DataLoader` instances for the MNIST dataset.
- **`ann_pytorch_model.py`**: Defines a simple ANN architecture (e.g., one hidden layer) using `torch.nn.Module`. Implements the `forward` pass. Includes a complete training loop:
  - Iterates through epochs and batches.
  - Calculates loss (e.g., `nn.CrossEntropyLoss`).
  - Performs backpropagation (`loss.backward()`).
  - Updates weights using an optimizer (e.g., `optim.Adam`).
  - Includes an evaluation function to test the model's accuracy on the test set.
  - Automatically detects and uses GPU if available (`torch.device`).

### Part D: Train on GPU (CUDA)

- **Purpose**: Explicitly leverage a CUDA-enabled GPU for faster training.
- **`ann_pytorch_cuda.py`**: Adapts the code from Part C to _explicitly_ run on my GPU.
  - Checks for CUDA availability (`torch.cuda.is_available()`).
  - Sets the device to `'cuda:0'`.
  - Moves the model (`.to(device)`) and data tensors (`.to(device)`) to the GPU during training and evaluation.
  - May include minor optimizations like `non_blocking=True` for data transfers and increased `num_workers` in `DataLoader`.

## ⚡️ Using GPU (CUDA) with PyTorch

By default, installing from `requirements.txt` will install the CPU-only version of PyTorch. If you want to use your NVIDIA GPU for training (recommended for faster training), you must install the CUDA-enabled version of PyTorch that matches your system.

### 1. Uninstall the CPU-only version (optional but recommended):
```bash
pip uninstall torch torchvision torchaudio
```

### 2. Install the correct CUDA-enabled version
Go to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) and select:
- Your OS
- Package: pip
- Language: Python
- Compute Platform: CUDA (choose the version that matches your system, e.g., CUDA 12.1)

Copy the provided install command. For example, for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verify CUDA is available
In Python, run:
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
```

If you see `True`, your environment is ready for GPU training!

## Requirements

The necessary Python libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

Each Python script (`neuron_*.py`, `ann_*.py`) can be run directly to see example usage or perform the specific task (like training):

```bash
python neuron_manual.py
python neuron_numpy.py
python ann_numpy_layer.py
python ann_pytorch_model.py  # Trains on CPU or GPU if available
python ann_pytorch_cuda.py   # Attempts to train explicitly on GPU
```

Training the PyTorch models (`ann_pytorch_model.py`, `ann_pytorch_cuda.py`) will download the MNIST dataset (if not already present in `./data`) and print training progress and final test accuracy.

See `reflection.txt` for my thoughts and learnings of this first step.