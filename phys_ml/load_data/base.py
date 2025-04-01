from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import torch
from torch.utils.data import Dataset

from ..config import Config


class FilebasedDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, config: Config):
        """
        Base-class for datasets. Based on the `torch.utils.data.Dataset`-class 
        but includes method to load data from disk.

        Parameters
        ----------
        config : Config
            A Config instance.
        """
        self.config = config
    
    @staticmethod
    @abstractmethod
    def load_from_file(path: str) -> torch.Tensor:
        """
        Load data from a given file-path.
        Overwrite this method in a dervied class to use it.

        Parameters
        ----------
        path : str
            File-path to load data from.

        Returns
        -------
        torch.Tensor
            Data as `torch.Tensor`.
        """
        pass


class SimpleDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: list[Any]|None = None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        if targets is not None:
            if isinstance(targets[0], int):
                self.targets = torch.tensor(targets, dtype=torch.long)
            else:
                self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, 'targets'):
            return self.inputs[idx], self.targets[idx]
        else:
            return self.inputs[idx]
