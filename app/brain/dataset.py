"""PyTorch Dataset and DataLoader for AURA Intent Classification."""

from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader


class IntentDataset(Dataset):
    """Custom PyTorch Dataset for intent classification training data.

    In PyTorch:
    - A Dataset represents a collection of (input_sample, target_label) pairs.
    - X: Numerical input feature vectors (Bag-of-Words vectors).
    - y: Ground truth target class indices (integers from 0 to num_classes - 1).
    """

    def __init__(
        self,
        x_data: Union[List[List[float]], torch.Tensor],
        y_data: Union[List[int], torch.Tensor],
    ) -> None:
        """Initialize the dataset with features and target labels.

        Args:
            x_data: 2D feature array or tensor of shape (N, vocab_size).
            y_data: 1D array or tensor of class integer indices of shape (N,).
        """
        if isinstance(x_data, torch.Tensor):
            self.x_data = x_data.clone().detach().float()
        else:
            self.x_data = torch.tensor(x_data, dtype=torch.float32)

        if isinstance(y_data, torch.Tensor):
            self.y_data = y_data.clone().detach().long()
        else:
            self.y_data = torch.tensor(y_data, dtype=torch.long)

        self.n_samples = len(self.x_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (input, target) sample by index.

        Args:
            index: Sample index.

        Returns:
            Tuple of (feature_tensor, label_tensor).
        """
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        """Return the total number of training samples in the dataset."""
        return self.n_samples


def get_data_loader(
    dataset: IntentDataset,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader to batch and shuffle training samples.

    Args:
        dataset: An instance of IntentDataset.
        batch_size: Number of samples per training batch.
        shuffle: Whether to randomize sample order at every epoch.

    Returns:
        A configured PyTorch DataLoader.
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
