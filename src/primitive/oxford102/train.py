import json
from datetime import datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from primitive.oxford102.model import SmallCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Logging setup
# ----------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = LOGS_DIR / f"run_{timestamp}.jsonl"


def log_event(event: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ----------------------------
# Dataloaders
# ----------------------------
IMG_SIZE = 224


def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.Flowers102(
        root="./dataset",
        split="train",
        download=True,
        transform=transform
    )
    valid_dataset = datasets.Flowers102(
        root="./dataset",
        split="val",
        download=True,
        transform=transform
    )
    test_dataset = datasets.Flowers102(
        root="./dataset",
        split="test",
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader


# ----------------------------
# Training
# ----------------------------
def train_model(epochs: int, batch_size: int, lr: float, export_path: Path):
    train_loader, valid_loader, _ = get_dataloaders(batch_size=batch_size)
    num_classes = 102

    model = SmallCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f"[Epoch {epoch + 1}/{epochs}] Validation Accuracy: {acc*100:.2f}%")

        # log result
        log_event({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": acc,
            "device": str(DEVICE),
            "timestamp": datetime.now().isoformat(),
        })

    torch.save(model.state_dict(), export_path)
    log_event({
        "event": "model_export",
        "export_path": str(export_path),
        "timestamp": datetime.now().isoformat(),
    })
    print(f"✅ Model exported to {export_path}")


# ----------------------------
# Testing
# ----------------------------
def test_model(model_path: Path, batch_size: int):
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    num_classes = 102

    model = SmallCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"✅ Loaded model from {model_path}")
    print("Running evaluation on test set...")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix shape:", cm.shape)

    log_event({
        "event": "test_complete",
        "model_path": str(model_path),
        "accuracy": acc,
        "num_samples": len(all_labels),
        "timestamp": datetime.now().isoformat(),
    })


# ----------------------------
# CLI
# ----------------------------
@click.group()
def cli():
    """Oxford 102 Flower CLI (train/test/export with logs)."""


@cli.command()
@click.option("--epochs", default=2, help="Number of epochs to train")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--export-path", default="model.pth", type=click.Path(), help="Where to save the model")
def train(epochs, batch_size, lr, export_path):
    """Train and export a new model."""
    train_model(epochs, batch_size, lr, Path(export_path))


@cli.command()
@click.option("--model-path", required=True, type=click.Path(exists=True), help="Path to exported model")
@click.option("--batch-size", default=32, help="Batch size")
def test(model_path, batch_size):
    """Test an exported model on the test set."""
    test_model(Path(model_path), batch_size)


if __name__ == "__main__":
    cli()
