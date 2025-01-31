import glob
import random

from copy import deepcopy

import h5py
import numpy as np

import torch

from tqdm import tqdm

from ..config.vertex import *
from . import FilebasedDataset


class AutoEncoderVertexDataset(FilebasedDataset):
    # matrix parameters
    n_freq = 24
    space_dim = 2
    k_dim = 3
    dim = k_dim
    length = n_freq**space_dim
    
    def __init__(self, config: VertexConfig):
        config.matrix_dim = self.dim
        self.data_in_indices: torch.Tensor = torch.tensor([])
        self.data_in_slices: torch.Tensor = torch.tensor([])

        # Subsample files
        file_paths = glob.glob(f"{config.path_train}/*.h5")
        subset = config.subset
        if subset is not None and subset != 0:
            n_files = len(file_paths)
            if type(subset) == float:
                subset = int(n_files * subset)
            if subset < n_files:
                if subset < 0:
                    subset = n_files + subset
                if config.subset_shuffle:
                    random.seed(config.subset_seed)
                    file_paths = random.sample(file_paths, max(subset, 1))
                else:
                    file_paths = file_paths[:subset]
        
        # Iterate through all files in given directory
        for file_path in file_paths:
            # Get vertex and create slices in each of the 3 dimensions
            vertex = self.load_from_file(file_path)

            # sample random indices of a 576^3 matrix and merge all rows through the sampled indices
            random.seed(config.sample_seed)
            merged_slices, indices = self.sample(vertex, config.sample_count_per_vertex)
        
            # Append result to data_in
            self.data_in_slices = torch.cat([self.data_in_slices, 
                                             torch.tensor(merged_slices, dtype=torch.float32)], axis=0)
            self.data_in_indices = torch.cat([self.data_in_indices, 
                                              torch.tensor(indices, dtype=torch.float32)], axis=0)
            assert self.data_in_indices.shape[0] == self.data_in_slices.shape[0]
        
        # Construct target data
        axis = config.construction_axis
        assert axis <= self.dim, f"Axis must be in range [1,{self.dim}]"
        idx_range = slice(self.length * (self.dim - axis), self.length * (self.dim - axis + 1))
        self.data_target = deepcopy(self.data_in_slices[:, idx_range])
        assert list(self.data_target[0]) == list(self.data_in_slices[0][idx_range])
    
    @classmethod
    def sample(cls, vertex: np.ndarray, sample_count_per_vertex: int) -> tuple[list[list[float]], np.ndarray]:
        indices = random.sample(range(cls.length**cls.dim), sample_count_per_vertex)
        indices = np.array([[(x // cls.length**i) % cls.length for i in range(cls.dim)] for x in indices])

        # Create and merge all row combinations
        merged_slices = [cls.get_vector_from_vertex(vertex, *idcs) for idcs in indices]
        return merged_slices, indices
    
    @staticmethod
    def get_vector_from_vertex(vertex: np.ndarray, x: int, y: int, z: int) -> list[float]:
        return [
            *vertex[x, y, :], 
            *vertex[x, :, z], 
            *vertex[:, y, z],
        ]
    
    """
    @classmethod
    def get_vector_from_vertex(cls, vertex: np.ndarray, *coord: int) -> np.ndarray:
        assert len(coord) == cls.dim, f'{cls.dim} coordinates required'
        vector = []
        for i in reversed(range(cls.dim)):  # select a full axis from vertex for each axis (dimension)
            c = [*coord]                    # choose a point in vertex: e.g. coords = [x, y, z]
            c[i] = slice(None)              # select a full axis: e.g. c = [x, y, :] for axis 1, c = [:, y, z] for axis 3
            vector.extend(vertex[c])        # select coordinates c from vertex and extend vector with selected axis
        return np.array(vector)
    """

    def __len__(self):
        return self.data_in_slices.shape[0]

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      return self.data_in_indices[idx], self.data_in_slices[idx], self.data_target[idx]

    @staticmethod
    def load_from_file(path: str) -> np.ndarray:
        with h5py.File(path, 'r') as f:
            for name, data in f["V"].items():
                if name.startswith("step"):
                    return data[()]
    
    @classmethod
    def to_6d_vertex(cls, vertex: np.ndarray) -> np.ndarray:
        return vertex.reshape((cls.n_freq,) * cls.space_dim * cls.k_dim)


class AutoEncoderVertex24x6Dataset(AutoEncoderVertexDataset):
    def __new__(cls, *args, **kwargs):
        cls.dim = cls.space_dim * cls.k_dim
        cls.length = cls.n_freq
        return super().__new__(cls)
    
    def __init__(self, config):
        super().__init__(config)
    
    @classmethod
    def sample(cls, vertex: np.ndarray, sample_count_per_vertex: int) -> tuple[list[list[float]], np.ndarray]:
        cls.__new__(cls)
        return super().sample(vertex, sample_count_per_vertex)
    
    @staticmethod
    def get_vector_from_vertex(vertex: np.ndarray, k1x: int, k1y: int, k2x: int, k2y: int, 
                               k3x: int, k3y: int) -> list[float]:
        return [
            *vertex[k1x, k1y, k2x, k2y, k3x, :],  # k3y
            *vertex[k1x, k1y, k2x, k2y, :, k3x],  # k3x
            *vertex[k1x, k1y, k2x, :, k3x, k3y],  # k2x
            *vertex[k1x, k1y, :, k2y, k3x, k3y],  # k2y
            *vertex[k1x, :, k2y, k3x, k3y, k3y],  # k1y
            *vertex[:, k1y, k2x, k3x, k3y, k3y],  # k1x
        ]
    
    @classmethod
    def to_3d_vertex(cls, vertex: np.ndarray) -> np.ndarray:
        return vertex.reshape((AutoEncoderVertexDataset.length,) * cls.k_dim, order='F')


class PredictVertexDataset(AutoEncoderVertexDataset):
    def __init__(self, config: VertexConfig, vertex_path: str|None = None, vertex: np.ndarray|None = None):
        assert vertex_path or vertex is not None, "Either vertex_path or vertex must be provided."
        if vertex is None:
            vertex = self.load_from_file(vertex_path)
        self.vertex = vertex
        self.config = config
        self.axis = self.config.construction_axis
        self.random_idx_generator = random.Random(config.sample_seed)

    def _get_random_idx(self) -> int:
        return self.random_idx_generator.randint(0, self.length - 1)
    
    def _create_input_vector(self, idcs: list[int]) -> torch.Tensor:
        inputs = torch.tensor(self.get_vector_from_vertex(self.vertex, *idcs), dtype=torch.float32)
        if self.config.positional_encoding:
            inputs = (torch.tensor(idcs), inputs)
        return inputs
    
    def _get_initial_indices(self, idx: int) -> tuple[list[int], int]:
        idcs = [(idx // self.length**(i - 1)) % self.length for i in range(self.dim - 1, 0, -1)]
        random_idx = self._get_random_idx()
        return idcs, random_idx

    def __len__(self):
        return self.length ** (self.dim - 1)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, list[int]]:
        idcs, random_idx = self._get_initial_indices(idx)
        idcs.insert(self.axis - 1, random_idx)
        inputs = self._create_input_vector(idcs)
        return inputs, idcs


class PredictVertex24x6Dataset(PredictVertexDataset, AutoEncoderVertex24x6Dataset):
    def __init__(self, config: VertexConfig, vertex_path: str|None = None, vertex: np.ndarray|None = None, 
                 dim: int|None = None, other_k: int|None = None, fixed_idcs: list[int]|None = None):
        PredictVertexDataset.__init__(self, config, vertex_path, vertex)
        if dim is not None:
            self.dim = dim
        self.other_k = other_k
        self.fixed_idcs = fixed_idcs

        if self.dim in [2, 3, 6]:
            self.replace_at = (self.axis - 1) % self.dim
        elif self.dim == 4:
            assert other_k is not None, "`other_k` must be provided for 4-dimensional vertex slice"
            k = (self.axis + 1) // 2
            ins_other = (other_k > k) * 2
            self.replace_at = 2 - ins_other + ((self.axis - 1) % 2)
        else:
            raise ValueError(f"Invalid value for `dim`: {self.dim}")
    
    def __len__(self):
        return PredictVertexDataset.__len__(self)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, list[int]]:
        idcs, random_idx = self._get_initial_indices(idx)
        if self.dim in [3, 6]:
            idcs.insert(self.axis - 1, random_idx)
            slice_idcs = idcs
        elif self.dim in [2, 4]:
            x = np.array([None] * 6)
            x[self.axis - 1] = 0
            x[(self.axis - 1) // 2 * 2 + self.axis % 2] = idcs[0]
            if self.dim == 4:
                x[(self.other_k - 1) * 2] = idcs[1]
                x[(self.other_k - 1) * 2 + 1] = idcs[2]
            slice_idcs = x[x != None]
            x[x== None] = self.fixed_idcs
            idcs = x
            slice_idcs = slice_idcs.tolist()
        inputs = self._create_input_vector(idcs)
        return inputs, slice_idcs


# run with:
# ```
# cd <...>/PhysML
# python -c "from phys_ml.load_data import vertex;vertex.convert_3d_to_6d_vertex('../frgs')"
# ```
def convert_3d_to_6d_vertex(data_dir: str) -> None:
    import os
    from pathlib import Path
    
    n_freq, dim = AutoEncoderVertexDataset.n_freq, 6

    data_dir: Path = Path(data_dir)
    new_dir = data_dir.parent / (data_dir.name + '_6d')
    os.makedirs(new_dir, exist_ok=True)

    file_paths = glob.glob(f"{data_dir}/*.h5")
    with tqdm(total=len(file_paths)) as prog:
        for file_path in file_paths:
            # load 3-dimensional vertex matrix
            vertex3 = AutoEncoderVertexDataset.load_from_file(file_path)

            # reshape to a 24^6 matrix
            vertex6 = vertex3.reshape((n_freq,) * dim)
            
            # store 6-dimesnional vertex matrix to disk
            file_name = Path(file_path).name
            with h5py.File(new_dir / file_name, 'w') as file:
                file.create_dataset("V/step0", data=vertex6, compression='lzf')
            
            prog.update()
