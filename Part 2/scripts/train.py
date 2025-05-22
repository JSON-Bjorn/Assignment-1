#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
import json
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(__file__))
from model import SimpleCNN
from data_augmentation import get_train_transforms, get_test_transforms
from utils import save_checkpoint, log_metrics, timer, log_timing


def plot_curves(metrics, output_dir):
    epochs = range(1, len(metrics["train_loss"]) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def show_and_save_examples(
    images, labels, preds, correct, output_dir, n=8, prefix="correct"
):
    idxs = np.where(correct)[0]
    if len(idxs) == 0:
        print(f"No {prefix} examples to show.")
        return
    n = min(n, len(idxs))
    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(idxs[:n]):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[idx].squeeze(), cmap="gray")
        plt.title(f"Pred: {preds[idx]}, True: {labels[idx]}")
        plt.axis("off")
    plt.suptitle(f"{prefix.capitalize()}ly Classified Examples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_examples.png"))
    plt.close()


def save_script_copy(output_dir):
    # Save a copy of this script for reproducibility
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(output_dir, "train_copy.py"))
    # If running in a notebook, note that as well
    with open(os.path.join(output_dir, "notebook_note.txt"), "w") as f:
        f.write(
            "If this run was performed in a notebook, please save a copy of the notebook here for reproducibility.\n"
        )


def main():
    # Unique output folder for this run
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "logs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hyperparameters (support environment variable overrides)
    def get_env_or_default(var, default, type_fn):
        v = os.environ.get(var)
        return type_fn(v) if v is not None else default

    training_config = {
        "batch_size": get_env_or_default("HP_BATCH_SIZE", 64, int),
        "lr": get_env_or_default("HP_LR", 0.001, float),
        "num_epochs": get_env_or_default("HP_NUM_EPOCHS", 20, int),
        "dropout_p": get_env_or_default("HP_DROPOUT_P", 0.5, float),
        "seed": get_env_or_default("HP_SEED", 42, int),
    }
    print("Training config:")
    print(json.dumps(training_config, indent=2))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    # Set random seeds for reproducibility
    torch.manual_seed(training_config["seed"])
    np.random.seed(training_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = datasets.MNIST(
        root="../../data", train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = datasets.MNIST(
        root="../../data", train=False, download=True, transform=get_test_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=training_config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_config["batch_size"], shuffle=False
    )

    # Model
    model = SimpleCNN(dropout_p=training_config["dropout_p"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

    best_val_acc = 0.0
    best_model_path = None
    timing_log = {}
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    epoch_times = []

    with timer("total_training", timing_log):
        start_time = datetime.now()
        for epoch in range(1, training_config["num_epochs"] + 1):
            epoch_start = datetime.now()
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            avg_train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct_train / total_train

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            avg_val_loss = val_loss / len(test_loader.dataset)
            val_acc = correct / total

            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(avg_val_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_acc"].append(val_acc)

            elapsed = datetime.now() - start_time
            print(
                f"Epoch {epoch:2d} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Time Elapsed={str(elapsed).split('.')[0]}"
            )

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(
                    checkpoint_dir, f"best_model_epoch{epoch}_valacc{val_acc:.4f}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                # Save config
                with open(
                    os.path.join(
                        checkpoint_dir, f"config_epoch{epoch}_valacc{val_acc:.4f}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(training_config, f, indent=2)

            epoch_times.append((epoch, str(datetime.now() - epoch_start).split(".")[0]))

    # Save metrics and timing
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "epoch_times.json"), "w") as f:
        json.dump(epoch_times, f, indent=2)
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(timing_log, f, indent=2)

    # Plot curves
    plot_curves(metrics, output_dir)

    # Reload best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_labels = []
    all_preds = []
    all_images = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_images.extend(images.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_images = np.array(all_images)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, output_dir)

    # Show/save correct and incorrect predictions
    correct_mask = all_labels == all_preds
    show_and_save_examples(
        all_images,
        all_labels,
        all_preds,
        correct_mask,
        output_dir,
        n=8,
        prefix="correct",
    )
    show_and_save_examples(
        all_images,
        all_labels,
        all_preds,
        ~correct_mask,
        output_dir,
        n=8,
        prefix="incorrect",
    )

    # Save a copy of the script and notebook note
    save_script_copy(output_dir)

    print(f"All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
