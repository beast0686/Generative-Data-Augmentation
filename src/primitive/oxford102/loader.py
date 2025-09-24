from functools import lru_cache
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dir = "dataset/train"
valid_dir = "dataset/valid"

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = list(Path(root_dir).glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path).convert("RGB") as image:
            if self.transform:
                image = self.transform(image)
            return image, str(img_path)

test_dataset = TestDataset(root_dir="dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


if __name__ == "__main__":
    for images, labels in train_loader:
        print("Train batch:", images.shape, labels.shape)
        break
    
    for images, paths in test_loader:
        print("Test batch:", images.shape, paths[:3])
        break
