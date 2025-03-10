from abc import ABC, abstractmethod

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

        """
        :param config: A Config instance.
        :type config: Config
        
        :param subset: Number of data items to load.
                       Either as integer to specify the absolute count or as float to specifiy the percentage of the existing data. 
                       Minimum items loaded is 1. (defaults to 1.0)
        :type subset: int | float, optional
        
        :param shuffle: If loading a subset, determines if items are selected from the directory in alphabetical order or randomly. 
                        (defaults to True)
        :type shuffle: bool, optional
        """
    
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
