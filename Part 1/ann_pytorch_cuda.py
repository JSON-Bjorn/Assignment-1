#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Import the MNIST loader function
from mnist_loader import get_mnist_loaders


# Re-use the same SimpleANN model definition
class SimpleANN(nn.Module):
    """A simple Artificial Neural Network using PyTorch nn.Module."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """Trains the PyTorch model on the specified device (GPU or CPU)."""
    model.train()
    print(f"\nStarting training on {device}...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            # Explicitly move data to the target device
            images = images.to(
                device, non_blocking=True
            )  # non_blocking for potential speedup
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                batch_loss = loss.item()
                batch_acc = (predicted == labels).sum().item() / images.size(0)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        epoch_duration = time.time() - epoch_start_time
        print(
            f"--- Epoch {epoch + 1} Finished --- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Duration: {epoch_duration:.2f}s ---"
        )

    total_training_time = time.time() - start_time
    print(f"Training finished. Total time: {total_training_time:.2f}s")


def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test dataset on the specified device."""
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    print(f"\nStarting evaluation on {device}...")
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = test_loss / total_samples
    accuracy = correct_predictions / total_samples
    eval_duration = time.time() - start_time
    print(f"Test Set Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    print(f"Evaluation Duration: {eval_duration:.2f}s")
    return accuracy


# Main execution block
if __name__ == "__main__":
    # --- Hyperparameters and Configuration ---
    INPUT_SIZE = 28 * 28
    HIDDEN_SIZE = 128
    NUM_CLASSES = 10
    BATCH_SIZE = 128  # Often larger batch sizes are better for GPUs
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    DATA_DIR = "./data"

    # --- Device Configuration (Prioritize CUDA) ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA is available. Using GPU.")
        # Optional: Print GPU details
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # --- Load Data ---
    # Use more workers if possible on GPU for faster data loading
    num_workers = 4 if device.type == "cuda" else 0
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, data_dir=DATA_DIR
    )
    print(f"MNIST data loaded successfully using {num_workers} workers.")

    # --- Initialize Model, Loss, Optimizer ---
    # Explicitly move the model to the target device
    model = SimpleANN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model, Loss function, and Optimizer initialized.")
    print(
        f"Model placed on: {next(model.parameters()).device}"
    )  # Verify model placement

    # --- Train the Model ---
    train_model(
        model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS
    )

    # --- Evaluate the Model ---
    evaluate_model(model, test_loader, criterion, device)

    # --- Optional: Save the trained model ---
    # torch.save(model.state_dict(), 'mnist_ann_model_cuda.pth')
    # print("\nModel state dictionary saved to mnist_ann_model_cuda.pth")
