import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report
from src.gan.gan import ConditionalGANAugmentor  # <-- conditional GAN
from datetime import datetime

# ----------------------------
# Settings
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS_CLASSIFIER = 50
EPOCHS_GAN = 50
IMG_SIZE = 224  # matches ResNet50 pretrained model input size

# ----------------------------
# Wrapper Dataset for consistent tensor types
# ----------------------------
class TensorDatasetWrapper(Dataset):
    """Wrapper to ensure consistent tensor types for both images and labels"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Ensure both image and label are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label

# ----------------------------
# Logging helper
# ----------------------------
def log_results(description, acc, y_true, y_pred):
    print(f"\n[{description}] Accuracy: {acc*100:.2f}%")
    print(f"[{description}] Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# ----------------------------
# Load Oxford 102 dataset
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.Flowers102(root="./dataset", split="train", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = datasets.Flowers102(root="./dataset", split="val", download=True, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = datasets.Flowers102(root="./dataset", split="test", download=True, transform=transform)

# ----------------------------
# Train conditional GAN
# ----------------------------
augmentor = ConditionalGANAugmentor(train_loader, img_size=IMG_SIZE, num_classes=102)
print("Training conditional GAN...")
augmentor.train_gan(epochs=EPOCHS_GAN)

# Generate synthetic dataset with proper labels
n_synthetic = 1000
synthetic_images, synthetic_labels = augmentor.generate_synthetic(n_samples=500)

# Ensure labels are long tensors
synthetic_labels = synthetic_labels.long()

synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=True)

# ----------------------------
# Combine real + synthetic
# ----------------------------
# Wrap both datasets to ensure consistent tensor types
wrapped_test_dataset = TensorDatasetWrapper(test_dataset)
combined_dataset = ConcatDataset([wrapped_test_dataset, synthetic_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# ----------------------------
# SimpleCNN Model (from try.py)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(SimpleCNN, self).__init__()
        # Adapted for 3-channel RGB images (flowers) instead of 1-channel (MNIST)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        # Adjusted for 224x224 input size instead of 28x28
        # After conv layers: (224-2)*(224-2) = 222*222, after pooling: 111*111
        # Then (111-2)*(111-2) = 109*109, after pooling: 54*54
        # Final size: 64 * 54 * 54 = 186624
        self.fc1 = nn.Linear(64 * 54 * 54, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# ----------------------------
# Initialize classifier
# ----------------------------
num_classes = 102
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # only train the final layer

# ----------------------------
# Classifier training
# ----------------------------
def train_classifier(model, loader, epochs=EPOCHS_CLASSIFIER):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

# ----------------------------
# Classifier testing
# ----------------------------
def test_classifier(model, loader, description="Test"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    log_results(description, acc, all_labels, all_preds)
    return acc

# ----------------------------
# Run pipeline
# ----------------------------
print("Training classifier on real + synthetic data...")
train_classifier(model, combined_loader, epochs=EPOCHS_CLASSIFIER)

print("\nEvaluating on validation set...")
test_classifier(model, valid_loader, description="Validation Set")