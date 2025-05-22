#! /usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime
import shutil
import time

# --- Output folder setup ---
run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
output_dir = os.path.join("Part 3", "logs", run_name)
os.makedirs(output_dir, exist_ok=True)

# --- Hyperparameters ---
hyperparams = {
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 5e-5,
    "img_size": 224,
    "num_classes": 4,
}

# Save hyperparameters as JSON
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(hyperparams, f, indent=2)
print("Hyperparameters:")
print(json.dumps(hyperparams, indent=2))

batch_size = hyperparams["batch_size"]
num_epochs = hyperparams["num_epochs"]
lr = hyperparams["lr"]
img_size = hyperparams["img_size"]
num_classes = hyperparams["num_classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved Data transforms
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Datasets
train_dir = os.path.join("Part 3", "Training")
test_dir = os.path.join("Part 3", "Testing")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# Optionally split train into train/val
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Compute class weights for balanced loss
labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(
    model.fc.in_features, num_classes
)  # Ensure output matches number of classes
for param in model.parameters():
    param.requires_grad = False
# Unfreeze the last block
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0
best_model_path = None
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_losses.append(running_loss / total)
    train_accs.append(correct / total)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)

    elapsed = time.time() - start_time
    epoch_time = time.time() - epoch_start
    print(
        f"Epoch {epoch + 1:2d} | Train Loss={train_losses[-1]:.4f} | Val Loss={val_losses[-1]:.4f} | Train Acc={train_accs[-1]:.4f} | Val Acc={val_accs[-1]:.4f} | Time Elapsed={int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
    )

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"transfer_checkpoint_epoch{epoch + 1}.pth"),
        )
    # Save best model
    if val_accs[-1] > best_val_acc:
        best_val_acc = val_accs[-1]
        best_model_path = os.path.join(output_dir, f"transfer_best_model.pth")
        torch.save(model.state_dict(), best_model_path)

# Plot loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(output_dir, "transfer_training_curves.png"))
plt.show()

# Test set evaluation with best model
model.load_state_dict(torch.load(best_model_path))
model.eval()
y_true, y_pred = [], []
all_images, all_labels, all_preds = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        all_images.extend(images.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "transfer_confusion_matrix.png"))
plt.show()

# Show/save correct and incorrect examples
all_images = np.array(all_images)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
correct_mask = all_labels == all_preds
incorrect_mask = ~correct_mask


def show_examples(images, labels, preds, mask, output_dir, n=8, prefix="correct"):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        print(f"No {prefix} examples to show.")
        return
    n = min(n, len(idxs))
    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(idxs[:n]):
        plt.subplot(1, n, i + 1)
        img = np.transpose(images[idx], (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array(
            [0.485, 0.456, 0.406]
        )  # unnormalize
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(
            f"Pred: {train_dataset.classes[preds[idx]]}\nTrue: {train_dataset.classes[labels[idx]]}"
        )
        plt.axis("off")
    plt.suptitle(f"{prefix.capitalize()}ly Classified Examples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_examples.png"))
    plt.close()


show_examples(
    all_images, all_labels, all_preds, correct_mask, output_dir, n=8, prefix="correct"
)
show_examples(
    all_images,
    all_labels,
    all_preds,
    incorrect_mask,
    output_dir,
    n=8,
    prefix="incorrect",
)

# Save a copy of the script
shutil.copy(__file__, os.path.join(output_dir, os.path.basename(__file__)))
