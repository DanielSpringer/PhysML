import random

from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from ..config.vertex import *
from ..load_data.vertex import *
from ..wrapper.vertex import *
from . import BaseTrainer


class VertexTrainer(BaseTrainer[VertexConfig, AutoEncoderVertexDataset, VertexWrapper]):
    @property
    def input_size(self) -> int:
        return self.dataset[0][1].shape[0]
    
    def pre_train(self) -> None:
        torch.set_float32_matmul_precision('high')
    
    def predict(self, vertex_path: str, new_vertex: np.ndarray|None = None, save_path: str = None, 
                load_model: bool = False, encode_only: bool = False):
        random.seed(42)
        return super().predict(vertex_path, new_vertex, save_path, load_model, 
                               encode_only=encode_only)
    
    def _predict_sample(self, vertex: torch.Tensor, coord: list[int], 
                        encode_only: bool = False) -> np.ndarray:
        device = self.get_device_from_accelerator(self.config.device_type)
        full_input = torch.tensor(self.dataset.get_vector_from_vertex(vertex, *coord),
                                  dtype=torch.float32).to(device)
        if self.config.positional_encoding:
            pos = torch.tensor(coord, dtype=torch.float32).to(device)
            full_input = (pos.unsqueeze(0), full_input.unsqueeze(0))
        if encode_only:
            pred = self.wrapper.model.encode(full_input).detach().cpu().numpy()
        else:
            pred = self.wrapper(full_input).detach().cpu().numpy()
        del full_input
        return pred
    
    def _predict(self, vertex: torch.Tensor, vertex_path: str, encode_only: bool = False) -> np.ndarray:
        axis = self.config.construction_axis
        result = self._prepare_prediction_matrix(vertex, axis, encode_only)
        with tqdm(total=self.dataset.length**(self.dataset.dim - 1)) as prog:
            for i_idx in range(self.dataset.length):
                for j_idx in range(self.dataset.length):
                    random_idx = random.randint(0, self.dataset.length - 1)
                    coord = [i_idx, j_idx]
                    coord.insert(axis - 1, random_idx)

                    # get prediction of length 576 for a k axis
                    pred = self._predict_sample(vertex, coord, encode_only)
                    
                    # feed back predictions into 576^3-matrix
                    coord[axis - 1] = slice(None)  # set the index for the prediction axis to the full axis
                    result[*coord] = pred          # assign the prediction to this axis
                    prog.update()
        
        # save results to disk
        subfolder = 'latentspaces' if encode_only else 'predictions'
        self._save_prediction(result, Path(vertex_path).stem, subfolder)
        return result
    
    def _prepare_prediction_matrix(self, vertex: torch.Tensor, axis: int = 3, 
                                   encode_only: bool = False) -> np.ndarray:
        if encode_only:
            shape = list(vertex.shape)
            shape[axis - 1] = self.config.hidden_dims[-1]
            result = np.empty(tuple(shape))
        else:
            result = np.zeros_like(vertex)
        return result

    def _save_prediction(self, vertex_prediction: np.ndarray, filename: str, subfolder: str):
        p = self.get_full_save_path() / subfolder
        p.mkdir(exist_ok=True)
        fp = p / f'{filename}.npy'
        np.save(fp, vertex_prediction)
    
    def load_latentspace(self, save_path: str|None = None, 
                         file_name: str|None = None) -> np.ndarray|None:
        return self._load_npy('latentspaces', save_path, file_name)


class VertexTrainer24x6(VertexTrainer):
    def __init__(self, project_name, config_name, subconfig_name = None, config_dir = 'configs', 
                 config_kwargs = ...):
        super().__init__(project_name, config_name, subconfig_name, config_dir, config_kwargs)
        self.dataset: AutoEncoderVertex24x6Dataset = self.dataset
    
    def predict(self, vertex_path: str, new_vertex: np.ndarray|None = None, save_path: str = None, 
                load_model: bool = False, encode_only: bool = False) -> np.ndarray:
        return super().predict(vertex_path, new_vertex, save_path, load_model, encode_only)

    def _predict(self, vertex: torch.Tensor, vertex_path: str, encode_only: bool = False) -> np.ndarray:
        axis = self.config.construction_axis
        result = self._prepare_prediction_matrix(vertex, axis, encode_only)
        with tqdm(total=self.dataset.n_freq**(self.dataset.dim - 1)) as prog:
            for a_idx in range(self.dataset.n_freq):
                for b_idx in range(self.dataset.n_freq):
                    for c_idx in range(self.dataset.n_freq):
                        for d_idx in range(self.dataset.n_freq):
                            for e_idx in range(self.dataset.n_freq):
                                random_idx = random.randint(0, self.dataset.n_freq - 1)
                                coord = [a_idx, b_idx, c_idx, d_idx, e_idx]
                                coord.insert(axis - 1, random_idx)  # on the current axis the random index is selected
                                
                                # get prediction of length 24 for an axis
                                pred = self._predict_sample(vertex, coord, encode_only)
                                
                                # feed back predictions into 24^6-matrix
                                coord[axis - 1] = slice(None)  # set the index for the prediction axis to the full axis
                                result[*coord] = pred          # assign the prediction to this axis
                                prog.update()
        
        # save results to disk
        subfolder = 'latentspaces' if encode_only else 'predictions'
        self._save_prediction(result, Path(vertex_path).stem, subfolder)
        return result

    def predict_3d(self, vertex_path: str, new_vertex: np.ndarray|None = None, save_path: str = None, 
                   load_model: bool = False, encode_only: bool = False) -> np.ndarray:
        pred = self.predict(vertex_path, new_vertex, save_path, load_model, encode_only)
        return self.dataset.to_3d_vertex(pred)
    
    def _prepare_slice_prediciton(self, vertex_path: str, new_vertex: np.ndarray|None = None, 
                                  save_path: str = None, load_model = False) -> tuple[torch.Tensor, int]:
        random.seed(42)
        axis = self.config.construction_axis
        new_data = super().prepare_prediciton(vertex_path, new_vertex, save_path, load_model)
        return new_data, axis
    
    def _prepare_slice_prediction_matrix(self, dim: int, encode_only: bool = False, 
                                         insert_at: int|None = None) -> np.ndarray:
        if encode_only:
            assert insert_at is not None, "If `encode_only` is True, `insert_at` must be provided."
            shape = (self.dataset.n_freq,) * dim
            shape[insert_at] = self.config.hidden_dims[-1]
            return np.zeros(shape)
        else:
            return np.zeros((self.dataset.n_freq,) * dim)

    def predict_slice2d(self, new_vertex_path: str, kix: int, kiy: int, kjx: int|None, kjy: int|None, 
                        new_vertex: np.ndarray|None = None, save_path: str|None = None, 
                        load_model: bool = False, encode_only: bool = False) -> np.ndarray:
        assert kix >= 1 and kix <= self.dataset.n_freq, f"kix must be in range [1,{self.dataset.n_freq}]"
        assert kiy >= 1 and kiy <= self.dataset.n_freq, f"kiy must be in range [1,{self.dataset.n_freq}]"
        assert kjx >= 1 and kjx <= self.dataset.n_freq, f"kjx must be in range [1,{self.dataset.n_freq}]"
        assert kjy >= 1 and kjy <= self.dataset.n_freq, f"kjy must be in range [1,{self.dataset.n_freq}]"
        
        vertex, axis = self._prepare_slice_prediciton(new_vertex_path, new_vertex, save_path, load_model)
        ins_at = (axis - 1) // 2 * 2
        pred_insert_idx = (axis - 1) % 2
        result = self._prepare_slice_prediction_matrix(dim=2, encode_only=encode_only, 
                                                       insert_at=pred_insert_idx)
        with tqdm(total=self.dataset.n_freq) as prog:
            for idx in range(self.dataset.n_freq):
                random_idx = random.randint(0, self.dataset.n_freq - 1)
                slice_coord = [idx, random_idx] if axis % 2 == 0 else [random_idx, idx]
                coord = [kix, kiy, kjx, kjy]
                coord.insert(ins_at, slice_coord[1])
                coord.insert(ins_at, slice_coord[0])
                
                # get prediction of length 24 for an axis
                pred = self._predict_sample(vertex, coord, encode_only)
                
                # feed back predictions into 24^6-matrix
                slice_coord[pred_insert_idx] = slice(None)  # set the index for the prediction axis to the full axis
                result[*slice_coord] = pred   # assign the prediction to this axis
                prog.update()

        filename = f'{Path(new_vertex_path).stem}_{kix}_{kiy}_{kjx}_{kjy}'
        subfolder = 'latentspace_slices' if encode_only else 'prediction_slices'
        self._save_prediction(result, filename, subfolder)
        return result
    
    def predict_slice4d(self, new_vertex_path: str, kix: int, kiy: int, other_k: int = 2,
                        new_vertex: np.ndarray|None = None, save_path: str|None = None, 
                        load_model: bool = False, encode_only: bool = False) -> np.ndarray:
        assert other_k >= 1 and other_k <= self.dataset.k_dim, f"`other_k` must be in range [1,{self.dataset.k_dim}]"
        assert other_k != ((self.config.construction_axis + 1) // 2), \
            f"`other_k` must be refer to a different k_i than `axis`."
        assert kix >= 1 and kix <= self.dataset.n_freq, f"`kix` must be in range [1,{self.dataset.n_freq}]"
        assert kiy >= 1 and kiy <= self.dataset.n_freq, f"`kiy` must be in range [1,{self.dataset.n_freq}]"
        

        vertex, axis = self._prepare_slice_prediciton(new_vertex_path, new_vertex, save_path, load_model)
        k = (axis + 1) // 2
        remaining_k = sum(range(1, self.dataset.k_dim + 1)) - k - other_k
        ins_remaining = (remaining_k - 1) * 2
        ins_other = (other_k > k) * 2
        a_idx_range = range(self.dataset.n_freq)
        b_idx_range = range(self.dataset.n_freq)
        pred_insert_idx = 2 - ins_other + ((axis - 1) % 2)
        result = self._prepare_slice_prediction_matrix(dim=4, encode_only=encode_only, 
                                                       insert_at=pred_insert_idx)
        with tqdm(total=self.dataset.n_freq**3) as prog:
            for a_idx in a_idx_range:
                for b_idx in b_idx_range:
                    for c_idx in range(self.dataset.n_freq):
                        random_idx = random.randint(0, self.dataset.n_freq - 1)
                        slice_coord = [c_idx, random_idx] if axis % 2 == 0 else [random_idx, c_idx]
                        slice_coord.insert(ins_other, b_idx)
                        slice_coord.insert(ins_other, a_idx)
                        coord = [c for c in slice_coord]
                        coord.insert(ins_remaining, kiy)
                        coord.insert(ins_remaining, kix)

                        # get prediction of length 24 for an axis
                        pred = self._predict_sample(vertex, coord, encode_only)
                        
                        # feed back predictions into 24^6-matrix
                        slice_coord[pred_insert_idx] = slice(None)  # set the index for the prediction axis to the full axis
                        result[*slice_coord] = pred   # assign the prediction to this axis
                        prog.update()
        
        filename = f'{Path(new_vertex_path).stem}_{kix}_{kiy}_k{other_k}'
        subfolder = 'latentspace_slices' if encode_only else 'prediction_slices'
        self._save_prediction(result, filename, subfolder)
        return result
