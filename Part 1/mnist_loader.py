import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """Loads and prepares the MNIST dataset.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_dir (str): The directory to download/load the MNIST data.

    Returns:
        tuple: A tuple containing the training DataLoader and test DataLoader.
    """
    # Define transformations for the dataset
    # 1. Convert images to PyTorch tensors
    # 2. Normalize the pixel values (mean=0.1307, std=0.3081 are standard for MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better learning
        num_workers=0 # Set to 0 for basic usage, increase for parallel loading if needed
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=0
    )

    return train_loader, test_loader

# Example Usage (Optional)
if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Get one batch of training images and labels
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(f"\nShape of a batch of images: {images.shape}") # [batch_size, channels, height, width]
    print(f"Shape of a batch of labels: {labels.shape}")   # [batch_size]

    # Get one batch of test images and labels
    test_dataiter = iter(test_loader)
    test_images, test_labels = next(test_dataiter)
    print(f"\nShape of a batch of test images: {test_images.shape}")
    print(f"Shape of a batch of test labels: {test_labels.shape}")