import glob
import os
import random

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import torch

from tqdm.notebook import tqdm

from ..config.vertex import *
from . import FilebasedDataset


class AutoEncoderVertexDataset(FilebasedDataset):
    # matrix parameters
    n_freq = 24
    space_dim = 2
    k_dim = 3
    phase_borders = {'afm': (0.0, 0.2), 'sc': (0.2, 0.33), 'fm': (0.33, 0.51)}

    @classmethod
    @property
    def dim(cls):
        return cls.k_dim
    
    @classmethod
    @property
    def length(cls):
        return cls.n_freq**cls.space_dim
    
    def __init__(self, config: VertexConfig, vertices: dict[str, np.ndarray]|None = None, file_paths: list[str]|None = None,
                 return_filepaths: bool = False):
        super().__init__(config)
        config.matrix_dim = self.dim
        self.return_filepaths = return_filepaths
        self.data_in_indices: torch.Tensor = torch.tensor([])
        self.data_in_slices: torch.Tensor = torch.tensor([])
        self.file_paths = None

        # Subsample files
        self.file_paths, config.subset = self.get_filepaths(config.path_train, config.subset, config.subset_shuffle, 
                                                            config.subset_seed, config.subset_type, file_paths)
        
        # Iterate through all files in given directory
        random.seed(config.sample_seed)
        for file_path in tqdm(self.file_paths, desc='Loading vertex data'):
            # Get vertex and create slices in each of the 3 dimensions
            if vertices:
                vertex = vertices[file_path]
            else:
                vertex = self.load_from_file(file_path)

            # sample random indices of a 576^3 matrix and merge all rows through the sampled indices
            merged_slices, indices = self.sample(vertex, config.sample_count_per_vertex, config=config)
        
            # Append result to data_in
            self.data_in_slices = torch.cat([self.data_in_slices, 
                                             torch.tensor(merged_slices, dtype=torch.float32)], dim=0)
            self.data_in_indices = torch.cat([self.data_in_indices, 
                                              torch.tensor(indices, dtype=torch.float32)], dim=0)
            assert self.data_in_indices.shape[0] == self.data_in_slices.shape[0]
        
        # Construct target data
        self.data_target = self.construct_targets()

    @classmethod
    def load_vertex_files(cls, file_paths: list[str]) -> dict[str, np.ndarray]:
        vertices = {}
        for file_path in tqdm(file_paths, desc='Loading vertex files'):
            vertex = cls.load_from_file(file_path)
            vertices[file_path] = vertex
        return vertices
    
    @classmethod
    def get_filepaths(cls, data_dir: str, subset: int|float|None, subset_shuffle: bool = True, subset_seed: int = 12,
                      subset_type: Literal['phase', 'sc', 'afm', 'fm']|None|list[str] = None, 
                      file_paths: list[str]|None = None) -> tuple[list[str], int]:
        # Subsample files
        random.seed(subset_seed)
        if file_paths is None:
            file_paths = [Path(fp).resolve().as_posix() for fp in glob.glob(f"{data_dir}/*.h5")]
        
        if subset_type and subset_type != 'phase':
            # select only vertices for certain phases
            if isinstance(subset_type, str):
                subset_type = [subset_type]
            fps = []
            for fp in file_paths:
                tp = float(Path(fp).stem[2:6])
                for phase in subset_type:
                    borders = cls.phase_borders[phase]
                    if tp >= borders[0] and tp < borders[1]:
                        fps.append(fp)
                        break
            file_paths = fps

        if subset is not None and subset != 0:
            if subset_type == 'phase':
                # select subset from each phase
                fps_by_phase = [[], [], []]
                for fp in file_paths:
                    tp = float(Path(fp).stem[2:6])
                    for i, phase in enumerate(['sc', 'afm', 'fm']):
                        borders = cls.phase_borders[phase]
                        if tp >= borders[0] and tp < borders[1]:
                            fps_by_phase[i].append(fp)
                            break
                
                file_paths = []
                for phase_fps in fps_by_phase:
                    n_files = len(phase_fps)
                    if type(subset) == float:
                        subset = int(round(n_files * subset, 0))
                    if subset < n_files:
                        if subset < 0:
                            subset = n_files + subset
                        fps = (random.sample(phase_fps, max(subset, 1)) if subset_shuffle 
                                else phase_fps[:subset])
                        file_paths.extend(fps)
                    else:
                        file_paths.extend(phase_fps)
            else:
                n_files = len(file_paths)
                if type(subset) == float:
                    subset = int(round(n_files * subset, 0))
                if subset < n_files:
                    if subset < 0:
                        subset = n_files + subset
                    file_paths = (random.sample(file_paths, max(subset, 1)) if subset_shuffle 
                                    else file_paths[:subset])
        return file_paths, subset
    
    @classmethod
    def sample(cls, vertex: np.ndarray, sample_count_per_vertex: int, 
               **kwargs) -> tuple[list[list[float]], np.ndarray]:
        indices = random.sample(range(cls.length**cls.dim), sample_count_per_vertex)
        indices = np.array([[(x // cls.length**i) % cls.length for i in range(cls.dim)] for x in indices])

        # Create and merge all row combinations
        merged_slices = [cls.get_vector_from_vertex(vertex, *idcs) for idcs in indices]
        return merged_slices, indices
    
    def construct_targets(self) -> torch.Tensor:
        axis = self.config.construction_axis
        assert axis <= self.dim, f"Axis must be in range [1,{self.dim}]"
        idx_range = slice(self.length * (self.dim - axis), self.length * (self.dim - axis + 1))
        targets = deepcopy(self.data_in_slices[:, idx_range])
        assert list(targets[0]) == list(self.data_in_slices[0][idx_range])
        return targets
    
    @classmethod
    def get_vector_from_vertex(cls, vertex: np.ndarray, x: int, y: int, z: int) -> list[float]:
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
        out = (self.data_in_slices[idx], self.data_in_indices[idx], self.data_target[idx])
        if self.return_filepaths:
            return *out, self.file_paths[idx // self.config.sample_count_per_vertex]
        return out

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
    @classmethod
    @property
    def dim(cls):
        return cls.space_dim * cls.k_dim
    
    @classmethod
    @property
    def length(cls):
        return cls.n_freq
    
    @classmethod
    def sample(cls, vertex: np.ndarray, sample_count_per_vertex: int, 
               **kwargs) -> tuple[list[list[float]], np.ndarray]:
        return super().sample(vertex, sample_count_per_vertex, **kwargs)
    
    @classmethod
    def get_vector_from_vertex(cls, vertex: np.ndarray, k1x: int, k1y: int, k2x: int, k2y: int, 
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


class AutoEncoder24x6InfoNCEDataset(AutoEncoderVertex24x6Dataset):
    def __init__(self, config: VertexConfig, vertex_dict: dict[str, np.ndarray]|None = None, file_paths: list[str]|None = None,
                 return_filepaths: bool = False):
        assert config.subset_type is None or len(config.subset_type) > 1, \
            f'Subset_type contains only {len(config.subset_type)} phases. Contrastive training is not possible with less than 2 phases.'
        FilebasedDataset.__init__(self, config)
        config.matrix_dim = self.dim
        self.return_filepaths = return_filepaths

        # Subsample files
        self.file_paths_by_phase, config.subset, n_fps = self.get_filepaths(config.path_train, config.subset, 
                                                                            config.subset_shuffle, config.subset_seed, 
                                                                            config.subset_type, file_paths)
        self.file_paths_by_phase = pd.DataFrame(self.file_paths_by_phase)
        
        # load and sample from vertices
        self.input_indices = torch.tensor([])
        self.input_vectors = torch.tensor([])
        self.file_paths = []
        self.n_phases = len(config.subset_type) if config.subset_type else 3
        for i, phases_fps in tqdm(self.file_paths_by_phase.iterrows(), desc='Loading vertex data', total=n_fps):
            # phases_fps <- 1 file path for each phase
            random.seed(config.sample_seed)
            
            # get 1 vertex for each phase
            if vertex_dict:
                vertices = [vertex_dict[fp] for fp in phases_fps]
            else:
                vertices = [self.load_from_file(fp) for fp in tqdm(phases_fps, desc='Loading files', leave=False)]
           
            # sample from vertices
            for i, vertex in tqdm(enumerate(vertices), desc='Sampling vertices', total=len(vertices), leave=False):
                # prepare a batch of samples containing the input, a matching sample and a negative sample for every other phase
                other_ids = [(i + j) % len(vertices) for j in range(1, self.n_phases)]
                other_vertices = [vertices[idx] for idx in other_ids]
                assert i not in other_ids, "Negative matches for vertex contain the vertex itself."
                # self.file_paths.extend(([phases_fps[i]] * 2 + [phases_fps[j] for j in other_ids]) * config.sample_count_per_vertex)
                
                input_samples, input_idcs = self.sample(vertex, config.sample_count_per_vertex)
                pos_idcs = input_idcs.copy()
                pos_idcs[:, -1] = (pos_idcs[:, -1] + 1) % self.n_freq
                pos_samples = [self.get_vector_from_vertex(vertex, *idcs) for idcs in pos_idcs]
                
                # construct negative examples
                neg_samples = [[self.get_vector_from_vertex(ov, *idcs) for idcs in input_idcs] for ov in other_vertices]

                # concatenate samples and indices cross-wise
                samples = np.array([input_samples, pos_samples, *neg_samples])
                idcs = np.array([input_idcs, pos_idcs, *([input_idcs] * (self.n_phases - 1))])
                samples = samples.transpose(1, 0, 2)
                idcs = idcs.transpose(1, 0, 2)
                # samples = np.concatenate(samples, axis=0)
                # idcs = np.concatenate(idcs, axis=0)
                assert samples.shape[-1] == self.n_freq * self.dim, \
                    f'Sample length should be {self.n_freq * self.dim} ' \
                    f'but is {samples.shape[-1]}'
                assert samples.shape[0] == config.sample_count_per_vertex, \
                    f'Number of samples should be {config.sample_count_per_vertex} samples per vertex, ' \
                    f'but is {samples.shape[0]}'
                assert samples.shape[1] == (self.n_phases + 1), \
                    f'Number of samples for contrastive loss should be {self.n_phases + 1} samples ' \
                    f'but is {samples.shape[1]}'
                self.input_vectors = torch.cat([self.input_vectors,
                                                torch.tensor(samples, dtype=torch.float32)], dim=0)
                self.input_indices = torch.cat([self.input_indices,
                                                torch.tensor(idcs, dtype=torch.float32)], dim=0)
                self.file_paths.extend([[phases_fps.iloc[i]] * 2 + [phases_fps.iloc[j] for j in other_ids]] * config.sample_count_per_vertex)
        assert self.input_vectors.shape[0] == self.input_indices.shape[0], \
            'Arrays of inputs and input-indices have different lengths.'
        self.targets = self.construct_targets(self.input_vectors)
    
    @classmethod
    def get_filepaths(cls, data_dir: str, subset: int|float|None, subset_shuffle: bool = True, subset_seed: int = 12,
                      subset_type: Literal['afm', 'sc', 'fm']|list[str]|None = None, 
                      file_paths: list[str]|None = None) -> tuple[dict[str, list[str]], int]:
        # Subsample files
        random.seed(subset_seed)
        if file_paths is None:
            file_paths = [Path(fp).resolve().as_posix() for fp in glob.glob(f"{data_dir}/*.h5")]
        if subset_type is None:
            subset_type = ['afm', 'sc', 'fm']
        
        fps_by_phase = {phase: [] for phase in subset_type}
        for fp in file_paths:
            tp = float(Path(fp).stem[2:6])
            for phase in fps_by_phase.keys():
                borders = cls.phase_borders[phase]
                if tp >= borders[0] and tp < borders[1]:
                    fps_by_phase[phase].append(fp)
                    break

        if subset is not None and subset != 0:
            # select subset of each phase
            for phase, fps in fps_by_phase.items():
                n_files = len(fps)
                phase_subset = subset
                if type(phase_subset) == float:
                    phase_subset = int(round(n_files * subset, 0))
                if phase_subset < n_files:
                    if phase_subset < 0:
                        phase_subset = n_files + phase_subset
                    fps = (random.sample(fps, max(phase_subset, 1)) if subset_shuffle 
                            else fps[:phase_subset])
                    fps_by_phase[phase] = fps
        
        # extend lists of file_paths to same length for each phase by random sampling 
        n_fps = max([len(fps) for fps in fps_by_phase.values()])
        for phase, fps in fps_by_phase.items():
            n = len(fps)
            if n < n_fps:
                fps_by_phase[phase] += random.choices(fps, k=(n_fps - n))
        return fps_by_phase, subset, n_fps
    
    @classmethod
    def sample(cls, vertex: np.ndarray, sample_count_per_vertex: int, 
               **kwargs) -> tuple[list[list[float]], np.ndarray]:
        indices = random.sample(range(cls.length**cls.dim), sample_count_per_vertex)
        indices = np.array([[(x // cls.length**i) % cls.length for i in range(cls.dim)] for x in indices])

        # Create and merge all row combinations
        merged_slices = [cls.get_vector_from_vertex(vertex, *idcs) for idcs in indices]
        return merged_slices, indices
    
    def construct_targets(self, input_vectors: torch.Tensor) -> torch.Tensor:
        axis = self.config.construction_axis
        assert axis <= self.dim, f"Axis must be in range [1,{self.dim}]"
        idx_range = slice(self.length * (self.dim - axis), self.length * (self.dim - axis + 1))
        targets = deepcopy(input_vectors[:, :, idx_range])
        assert list(targets[0, 0]) == list(input_vectors[0, 0, idx_range])
        return targets
    
    def __len__(self):
        return self.input_vectors.shape[0]

    def __getitem__(self, idx: int):
        out = (self.input_vectors[idx], self.input_indices[idx], self.targets[idx])
        if self.return_filepaths:
            i_vertex_sample = idx // self.config.sample_count_per_vertex
            i_phase_vertices = i_vertex_sample // self.n_phases
            i_phase = i_vertex_sample % self.n_phases
            i_phases = [i_phase] * 2 + [(i_phase + j) % self.n_phases for j in range(1, self.n_phases)]
            fps = self.file_paths_by_phase.iloc[i_phase_vertices, i_phases].tolist()  # filepaths for the sub-batch
            return *out, fps
        return out


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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
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
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
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
def convert_3d_to_6d_vertex(data_dir: str) -> np.ndarray:
    from tqdm import tqdm

    n_freq, dim = AutoEncoderVertexDataset.n_freq, 6
    data_dir: Path = Path(data_dir)
    new_dir = data_dir.parent / (data_dir.name + '_6d')
    os.makedirs(new_dir, exist_ok=True)

    file_paths = glob.glob(f"{data_dir}/*.h5")
    with tqdm(total=len(file_paths)) as prog:
        for file_path in file_paths:
            vertex = AutoEncoderVertexDataset.load_from_file(file_path)

            # reshape to a 24^6 matrix
            vertex_convert = vertex.reshape((n_freq,) * dim)
            
            # store new vertex to disk
            file_name = Path(file_path).name
            with h5py.File(new_dir / file_name, 'w') as file:
                file.create_dataset("V/step0", data=vertex_convert, compression='lzf')
            prog.update()
