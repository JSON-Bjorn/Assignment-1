#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import the MNIST loader function from the other file
from mnist_loader import get_mnist_loaders


class SimpleANN(nn.Module):
    """A simple Artificial Neural Network using PyTorch nn.Module."""

    def __init__(self, input_size, hidden_size, num_classes):
        """Initializes the ANN layers.

        Args:
            input_size (int): The number of input features (e.g., 28*28 for MNIST).
            hidden_size (int): The number of neurons in the hidden layer.
            num_classes (int): The number of output classes (e.g., 10 for MNIST digits).
        """
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # ReLU activation function
        self.layer2 = nn.Linear(hidden_size, num_classes)
        # Note: CrossEntropyLoss in PyTorch combines LogSoftmax and NLLLoss,
        # so we don't need a final activation layer here if using it.

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor (batch_size, num_classes).
        """
        x = self.flatten(x)  # Flatten the image
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """Trains the PyTorch model."""
    model.train()  # Set the model to training mode
    print(f"\nStarting training on {device}...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            # Move data to the specified device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:  # Print progress every 100 batches
                batch_loss = loss.item()
                batch_acc = (predicted == labels).sum().item() / images.size(0)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Batch Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(
            f"--- Epoch {epoch + 1} Finished --- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} ---"
        )

    print("Training finished.")


def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test dataset."""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    print(f"\nStarting evaluation on {device}...")

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = test_loss / total_samples
    accuracy = correct_predictions / total_samples
    print(f"Test Set Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    return accuracy


# Main execution block
if __name__ == "__main__":
    # --- Hyperparameters and Configuration ---
    INPUT_SIZE = 28 * 28  # MNIST image dimensions
    HIDDEN_SIZE = 128  # Number of neurons in the hidden layer
    NUM_CLASSES = 10  # Number of output classes (digits 0-9)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    DATA_DIR = "./data"  # Directory for MNIST dataset

    # --- Device Configuration (Force CPU) ---
    device = torch.device("cpu")
    print("Forcing CPU usage for this script.")

    # --- Load Data ---
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE, data_dir=DATA_DIR
    )
    print("MNIST data loaded successfully.")

    # --- Initialize Model, Loss, Optimizer ---
    model = SimpleANN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model, Loss function, and Optimizer initialized.")

    # --- Train the Model ---
    train_model(
        model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS
    )

    # --- Evaluate the Model ---
    evaluate_model(model, test_loader, criterion, device)

    # --- Optional: Save the trained model ---
    # torch.save(model.state_dict(), 'mnist_ann_model.pth')
    # print("\nModel state dictionary saved to mnist_ann_model.pth")
