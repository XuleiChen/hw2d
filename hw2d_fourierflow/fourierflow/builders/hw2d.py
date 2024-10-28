import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from .base import Builder

class HasegawaWakataniBuilder(Builder):
    name = 'hasegawa_wakatani'

    def __init__(self, train_data_path, valid_data_path, test_data_path, k, traj_len ,batch_size=32, num_workers=1, feature='density', **kwargs):
        """
        Builder for loading Hasegawa-Wakatani equation datasets.

        Parameters:
        - train_data_path: Path to the .h5 file containing the training dataset.
        - valid_data_path: Path to the .h5 file containing the validation dataset.
        - test_data_path: Path to the .h5 file containing the testing dataset.
        - k: Step size parameter for downsampling.
        - batch_size: Batch size for DataLoader.
        - num_workers: Number of workers for DataLoader.
        - feature: The variable to use ('density', 'omega', or 'phi').
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k
        self.traj_len=traj_len
        self.feature = feature
        self.kwargs = kwargs

        # Create datasets for train, validation, and test splits
        self.train_dataset = HWTorchDataset(train_data_path, k=self.k, feature=self.feature)
        self.valid_dataset = HWTrajectoryDataset(valid_data_path, traj_len=self.traj_len, feature=self.feature)
        self.test_dataset = HWTrajectoryDataset(test_data_path, traj_len=self.traj_len, feature=self.feature)

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers, **self.kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers, **self.kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers, **self.kwargs)
        return loader

    def inference_data(self):
        return self.test_dataloader()

class HWTorchDataset(Dataset):
    def __init__(self, data_path, k, feature='density', indices=None, mu_value=5e-08):
        """
        Dataset for training with single-step prediction.

        Parameters:
        - data_path: Path to the .h5 file containing the simulation data.
        - k: Step size parameter for downsampling.
        - feature: The variable to use ('density', 'omega', or 'phi').
        - indices: Optional list of indices to subset the dataset.
        - mu_value: Constant value for the 'mu' feature, default is 5e-08.
        """
        self.data_path = data_path
        self.k = k
        self.feature = feature
        self.mu_value = mu_value

        # Load the dataset from the .h5 file
        with h5py.File(data_path, 'r') as f:
            self.data = torch.tensor(f[feature][:], dtype=torch.float32)

        # If indices are provided, subset the data accordingly
        if indices is not None:
            self.data = self.data[indices]


    def __len__(self):
        # Return the length minus one to account for the shift in indices
        return (self.data.shape[0] - 1) // self.k

    def __getitem__(self, idx):
        # Compute the time index for x and y based on the batch index
        time_index_x = idx * self.k  # Corresponds to 0, k, 2k, ...
        time_index_y = time_index_x + 1  # Corresponds to 1, k+1, 2k+1, ...

        # Extract the corresponding 2D slices for x and y
        x = self.data[time_index_x,...].unsqueeze(-1)  # Shape: (X, Y, 1)
        y = self.data[time_index_y,...].unsqueeze(-1)  # Shape: (X, Y, 1)

        return {
            'x': x,  # Single 2D input at the specified time step (X, Y)
            'mu': torch.tensor(self.mu_value, dtype=torch.float32),  # Constant 'mu' value
            'y': y  # Single 2D target at the next time step (X, Y)
        }

class HWTrajectoryDataset(Dataset):
    def __init__(self, data_path, traj_len, feature='density', indices=None, mu_value=5e-08):
        """
        Dataset for validation/testing with multistep prediction.

        Parameters:
        - data_path: Path to the .h5 file containing the simulation data.
        - traj_len: The length of the trajectory to use for each sample.
        - feature: The variable to use ('density', 'omega', or 'phi').
        - indices: Optional list of indices to subset the dataset.
        - mu_value: Constant value for the 'mu' feature, default is 5e-08.
        """
        self.data_path = data_path
        self.traj_len = traj_len
        self.feature = feature
        self.mu_value = mu_value

        # Load the dataset from the .h5 file
        with h5py.File(data_path, 'r') as f:
            self.data = torch.tensor(f[feature], dtype=torch.float32)

        # If indices are provided, subset the data accordingly
        if indices is not None:
            self.data = self.data[indices]

        # Ensure we only use the part that can be evenly divided by traj_len
        num_steps = self.data.shape[0]
        self.num_steps = num_steps // traj_len  # Use the largest divisible part
        self.data = self.data[:self.num_steps * traj_len].view(self.num_steps, traj_len, *self.data.shape[1:])


    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        # Extract the trajectory data for the given index
        data_traj = self.data[idx]  # Shape: (traj_len, X, Y)
        # Permute to (X, Y, traj_len) to match the desired shape
        data_traj = data_traj.permute(1, 2, 0)
        times = torch.arange(0, self.traj_len)  # Shape: (traj_len,)
        return {
            'data': data_traj,  # Shape: (X, Y, traj_len)
            #'mu': torch.tensor(self.mu_value, dtype=torch.float32),  # Scalar 'mu' value
            #'times': times
        }