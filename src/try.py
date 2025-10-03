"""
Complete CNN + UNet2D Diffusion Pipeline for Flowers102 Classification
Single file implementation with comprehensive evaluation

This script:
1. Loads Flowers102 dataset limited to 500 samples
2. Trains a baseline CNN on original data
3. Trains a UNet2D diffusion model for data augmentation
4. Generates 500 synthetic flower images
5. Trains an augmented CNN on original + synthetic data
6. Compares performance and saves comprehensive metrics```
"""

import os
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import Flowers102

# Try to import diffusers - install if needed
try:
    from diffusers import DDPMScheduler, UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Install with: pip install diffusers")
    DIFFUSERS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    # Dataset parameters
    NUM_SAMPLES = 500  # Limit dataset to 500 samples
    NUM_CLASSES = 102
    IMAGE_SIZE = 128

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CNN_EPOCHS = 50
    DIFFUSION_EPOCHS = 5

    # Diffusion parameters
    DIFFUSION_BATCH_SIZE = 16
    DIFFUSION_LR = 1e-4
    NUM_TRAIN_TIMESTEPS = 1000
    NUM_INFERENCE_STEPS = 50
    SYNTHETIC_SAMPLES = 500

    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Output
    EXPERIMENT_NAME = "flowers102_unet2d_diffusion"
    SAVE_RESULTS = True


config = Config()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_directories():
    """Create necessary directories"""
    dirs = ["results", "models", "synthetic_data", "experiments"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)


def save_results(results, experiment_name):
    """Save experimental results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{experiment_name}_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    json_results = convert_for_json(results)

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {filename}")
    return filename


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_flowers102_limited(num_samples=500, data_dir="./dataset"):
    """Load Flowers102 dataset limited to specified number of samples"""
    print(f"Loading Flowers102 dataset (limited to {num_samples} samples)...")

    # Define transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load full dataset
    train_dataset_full = Flowers102(
        root=data_dir, split="train", transform=transform_train, download=True
    )
    val_dataset_full = Flowers102(
        root=data_dir, split="val", transform=transform_test, download=True
    )
    test_dataset_full = Flowers102(
        root=data_dir, split="test", transform=transform_test, download=True
    )

    # Create limited subset
    # Take samples proportionally from each split
    train_samples = int(num_samples * config.TRAIN_RATIO)
    val_samples = int(num_samples * config.VAL_RATIO)
    test_samples = num_samples - train_samples - val_samples

    # Create random indices for subsets
    train_indices = random.sample(
        range(len(train_dataset_full)), min(train_samples, len(train_dataset_full))
    )
    val_indices = random.sample(
        range(len(val_dataset_full)), min(val_samples, len(val_dataset_full))
    )
    test_indices = random.sample(
        range(len(test_dataset_full)), min(test_samples, len(test_dataset_full))
    )

    # Create subsets
    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(val_dataset_full, val_indices)
    test_subset = Subset(test_dataset_full, test_indices)

    print("Dataset loaded:")
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    print(f"  Test samples: {len(test_subset)}")

    return train_subset, val_subset, test_subset


# =============================================================================
# CNN MODEL DEFINITION
# =============================================================================


class FlowerCNN(nn.Module):
    """Enhanced CNN for flower classification"""

    def __init__(self, num_classes=102):
        super(FlowerCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# UNET2D DIFFUSION MODEL
# =============================================================================


def create_unet2d_model():
    """Create UNet2D model for diffusion"""
    if not DIFFUSERS_AVAILABLE:
        print("Error: diffusers library not available")
        return None

    model = UNet2DModel(
        sample_size=config.IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


class SyntheticFlowerDataset(Dataset):
    """Dataset for synthetic flower images"""

    def __init__(self, synthetic_images, labels, transform=None):
        self.synthetic_images = synthetic_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.synthetic_images)

    def __getitem__(self, idx):
        image = self.synthetic_images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_cnn(model, train_loader, val_loader, epochs, device, model_name="CNN"):
    """Train CNN model with validation tracking"""
    print(f"\n=== Training {model_name} ===")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update learning rate
        scheduler.step()

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/best_{model_name.lower()}.pth")

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:2d}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

    print(f"{model_name} training completed! Best Val Acc: {best_val_acc:.2f}%")
    return history


def train_diffusion_model(train_loader, device):
    """Train UNet2D diffusion model"""
    if not DIFFUSERS_AVAILABLE:
        print("Skipping diffusion training - diffusers not available")
        return None

    print("\n=== Training UNet2D Diffusion Model ===")

    model = create_unet2d_model().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.NUM_TRAIN_TIMESTEPS)

    optimizer = optim.AdamW(model.parameters(), lr=config.DIFFUSION_LR)

    model.train()
    for epoch in range(config.DIFFUSION_EPOCHS):
        total_loss = 0

        for step, (clean_images, _) in enumerate(train_loader):
            clean_images = clean_images.to(device)

            # Sample noise and timesteps
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=device,
            ).long()

            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % 10 == 0 or epoch == config.DIFFUSION_EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:2d}/{config.DIFFUSION_EPOCHS}: Loss: {avg_loss:.6f}"
            )

    # Save model
    torch.save(model.state_dict(), "models/diffusion_model.pth")
    print("Diffusion model training completed!")

    return model, noise_scheduler


def generate_synthetic_data(model, noise_scheduler, num_samples, device):
    """Generate synthetic flower images"""
    if not DIFFUSERS_AVAILABLE or model is None:
        print("Skipping synthetic data generation - diffusers not available")
        return None, None

    print(f"\n=== Generating {num_samples} Synthetic Samples ===")

    model.eval()
    synthetic_images = []
    synthetic_labels = []

    batch_size = config.DIFFUSION_BATCH_SIZE
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

            # Start with random noise
            sample = torch.randn(
                current_batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE
            ).to(device)

            # Denoising loop
            for t in noise_scheduler.timesteps:
                # Move timestep to device and create batch
                t_batch = t.unsqueeze(0).repeat(current_batch_size).to(device)

                # Predict noise
                noise_pred = model(sample, t_batch).sample

                # Remove noise
                sample = noise_scheduler.step(noise_pred, t, sample).prev_sample

            # Convert to images and assign random labels
            for i in range(current_batch_size):
                synthetic_images.append(sample[i].cpu())
                synthetic_labels.append(random.randint(0, config.NUM_CLASSES - 1))

            print(f"Generated batch {batch_idx + 1}/{num_batches}")

    print(
        f"Synthetic data generation completed! Generated {len(synthetic_images)} samples"
    )
    return synthetic_images, synthetic_labels


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate_model(model, test_loader, device):
    """Evaluate model and return comprehensive metrics"""
    print("\n=== Model Evaluation ===")

    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    # Additional metrics
    precision_weighted = precision_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        all_targets, all_predictions, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "test_loss": test_loss / len(test_loader),
        "num_samples": len(all_targets),
    }

    print("Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro): {recall:.4f}")
    print(f"  F1-Score (macro): {f1:.4f}")
    print(f"  F1-Score (weighted): {f1_weighted:.4f}")

    return metrics, all_predictions, all_targets


def compare_results(baseline_metrics, augmented_metrics):
    """Compare baseline vs augmented results"""
    print("\n=== Performance Comparison ===")
    print("-" * 80)
    print(f"{'Metric':<20} {'Baseline':<12} {'Augmented':<12} {'Improvement':<15}")
    print("-" * 80)

    improvements = {}
    key_metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]

    for metric in key_metrics:
        baseline_val = baseline_metrics[metric]
        augmented_val = augmented_metrics[metric]
        improvement = ((augmented_val - baseline_val) / baseline_val) * 100

        improvements[metric] = {
            "baseline": baseline_val,
            "augmented": augmented_val,
            "absolute_improvement": augmented_val - baseline_val,
            "relative_improvement_pct": improvement,
        }

        print(
            f"{metric.replace('_', ' ').title():<20} "
            f"{baseline_val:<12.4f} "
            f"{augmented_val:<12.4f} "
            f"{improvement:>+7.2f}%"
        )

    print("-" * 80)

    avg_improvement = np.mean(
        [improvements[m]["relative_improvement_pct"] for m in key_metrics]
    )
    print(f"Average Improvement: {avg_improvement:+.2f}%")

    return improvements


# =============================================================================
# MAIN EXPERIMENT PIPELINE
# =============================================================================


def main():
    """Main experiment pipeline"""
    print("=== Complete Flowers102 CNN + UNet2D Diffusion Experiment ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    set_seeds(42)
    device = get_device()
    create_directories()

    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Dataset samples: {config.NUM_SAMPLES}")
    print(f"  Synthetic samples: {config.SYNTHETIC_SAMPLES}")
    print(f"  Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  CNN epochs: {config.CNN_EPOCHS}")
    print(f"  Diffusion epochs: {config.DIFFUSION_EPOCHS}")
    print()

    # Load data
    train_dataset, val_dataset, test_dataset = load_flowers102_limited(
        config.NUM_SAMPLES
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Diffusion data loader (smaller batch size)
    diffusion_train_loader = DataLoader(
        train_dataset, batch_size=config.DIFFUSION_BATCH_SIZE, shuffle=True
    )

    # 1. Train baseline CNN
    print("\n" + "=" * 60)
    print("STEP 1: BASELINE CNN TRAINING")
    print("=" * 60)

    baseline_cnn = FlowerCNN(config.NUM_CLASSES).to(device)
    baseline_history = train_cnn(
        baseline_cnn,
        train_loader,
        val_loader,
        config.CNN_EPOCHS,
        device,
        "Baseline_CNN",
    )

    # Evaluate baseline
    baseline_metrics, baseline_preds, baseline_targets = evaluate_model(
        baseline_cnn, test_loader, device
    )

    # 2. Train diffusion model and generate synthetic data
    print("\n" + "=" * 60)
    print("STEP 2: DIFFUSION MODEL TRAINING & GENERATION")
    print("=" * 60)

    diffusion_model, noise_scheduler = train_diffusion_model(
        diffusion_train_loader, device
    )
    synthetic_images, synthetic_labels = generate_synthetic_data(
        diffusion_model, noise_scheduler, config.SYNTHETIC_SAMPLES, device
    )

    # 3. Train augmented CNN
    print("\n" + "=" * 60)
    print("STEP 3: AUGMENTED CNN TRAINING")
    print("=" * 60)

    if synthetic_images is not None:
        # Create augmented dataset
        transform_synthetic = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        synthetic_dataset = SyntheticFlowerDataset(
            synthetic_images, synthetic_labels, transform=transform_synthetic
        )

        # Combine datasets
        from torch.utils.data import ConcatDataset

        augmented_train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

        augmented_train_loader = DataLoader(
            augmented_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )

        # Train augmented CNN
        augmented_cnn = FlowerCNN(config.NUM_CLASSES).to(device)
        augmented_history = train_cnn(
            augmented_cnn,
            augmented_train_loader,
            val_loader,
            config.CNN_EPOCHS,
            device,
            "Augmented_CNN",
        )

        # Evaluate augmented model
        augmented_metrics, augmented_preds, augmented_targets = evaluate_model(
            augmented_cnn, test_loader, device
        )

        # Compare results
        improvements = compare_results(baseline_metrics, augmented_metrics)
    else:
        print("Skipping augmented training due to diffusion model issues")
        augmented_metrics = baseline_metrics
        improvements = {}

    # 4. Save comprehensive results
    print("\n" + "=" * 60)
    print("STEP 4: SAVING RESULTS")
    print("=" * 60)

    results = {
        "experiment_info": {
            "name": config.EXPERIMENT_NAME,
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "num_original_samples": config.NUM_SAMPLES,
            "num_synthetic_samples": config.SYNTHETIC_SAMPLES
            if synthetic_images
            else 0,
            "image_size": config.IMAGE_SIZE,
            "num_classes": config.NUM_CLASSES,
        },
        "dataset_info": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "synthetic_samples": len(synthetic_images) if synthetic_images else 0,
        },
        "training_config": {
            "cnn_epochs": config.CNN_EPOCHS,
            "diffusion_epochs": config.DIFFUSION_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "diffusion_lr": config.DIFFUSION_LR,
        },
        "baseline_results": {
            "metrics": baseline_metrics,
            "training_history": baseline_history,
        },
        "augmented_results": {
            "metrics": augmented_metrics,
            "training_history": augmented_history if synthetic_images else None,
        },
        "improvements": improvements,
        "diffusers_available": DIFFUSERS_AVAILABLE,
    }

    if config.SAVE_RESULTS:
        results_file = save_results(results, config.EXPERIMENT_NAME)

        # Save metrics comparison CSV
        metrics_df = pd.DataFrame(
            {
                "Model": ["Baseline", "Augmented"],
                "Accuracy": [
                    baseline_metrics["accuracy"],
                    augmented_metrics["accuracy"],
                ],
                "Precision": [
                    baseline_metrics["precision_macro"],
                    augmented_metrics["precision_macro"],
                ],
                "Recall": [
                    baseline_metrics["recall_macro"],
                    augmented_metrics["recall_macro"],
                ],
                "F1-Score": [
                    baseline_metrics["f1_macro"],
                    augmented_metrics["f1_macro"],
                ],
                "Training_Samples": [
                    len(train_dataset),
                    len(train_dataset)
                    + (len(synthetic_images) if synthetic_images else 0),
                ],
            }
        )

        csv_file = results_file.replace(".json", "_metrics.csv")
        metrics_df.to_csv(csv_file, index=False)
        print(f"Metrics CSV saved to: {csv_file}")

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Augmented Accuracy: {augmented_metrics['accuracy']:.4f}")
    if improvements:
        acc_improvement = improvements["accuracy"]["relative_improvement_pct"]
        print(f"Improvement: {acc_improvement:+.2f}%")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()