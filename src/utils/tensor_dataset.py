import torch
from torch.utils.data import Dataset



class TensorDatasetWrapper(Dataset):
    """Wrapper to ensure consistent tensor types for both images and labels"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label
