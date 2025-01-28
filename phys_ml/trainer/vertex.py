from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..config.vertex import *
from ..load_data.vertex import *
from ..wrapper.vertex import *
from . import BaseTrainer, CKPT_TYPE, TrainerModes


class VertexTrainer(BaseTrainer[VertexConfig, AutoEncoderVertexDataset, VertexWrapper]):
    def __init__(self, project_name: str, config_name: str | None = None, 
                 subconfig_name: str|None = None, load_from: str|None = None, config_dir: str = 'configs', 
                 config_kwargs: dict[str, Any] = {}):
        super().__init__(project_name, config_name, subconfig_name, load_from, config_dir, 
                         config_kwargs)
        torch.set_float32_matmul_precision('high')
    
    @property
    def input_size(self) -> int:
        return self.dataset[0][1].shape[0]
    
    def predict(self, vertex_path: str, new_vertex: np.ndarray|None = None, train_mode: TrainerModes|None = None, 
                load_from: Literal['best', 'last']|str|None = None, encode_only: bool = False):
        dataset = self.config.predict_dataset(self.config, vertex_path, new_vertex)
        dataloader = self.data_loader(dataset, batch_size=self.config.batch_size, 
                                      num_workers=self.config.num_dataloader_workers, 
                                      persistent_workers=bool(self.config.num_dataloader_workers),
                                      pin_memory=True)

        # predict
        if load_from:
            ckpt_path = self.get_model_ckpt(load_from)
            self.init_trainer(train_mode, Path(ckpt_path).parents[1])
        pred_vertex = self.prepare_prediction_matrix(dataset.dim, encode_only, 
                                                     replace_at=self.config.construction_axis-1)
        self.wrapper.set_predictor(pred_vertex, encode_only)
        self.trainer.predict(self.wrapper, dataloader, return_predictions=False, ckpt_path=ckpt_path)
        
        # save results to disk
        pred_vertex = self.wrapper.pred_vertex.cpu().numpy()
        subfolder = 'latentspaces' if encode_only else 'predictions'
        self.save_prediction(pred_vertex, Path(vertex_path).stem, subfolder)
        return pred_vertex
    
    def prepare_prediction_matrix(self, out_dim: int, encode_only: bool = False, 
                                   replace_at: int|None = None) -> torch.Tensor:
        #device = self.get_device_from_accelerator(self.config.device_type)
        if encode_only:
            assert replace_at is not None, "If `encode_only` is True, `insert_at` must be provided."
            shape = [self.dataset.length] * out_dim
            shape[replace_at] = self.config.hidden_dims[-1]
            pred_vertex = np.zeros(tuple(shape))
        else:
            pred_vertex = np.zeros((self.dataset.length,) * out_dim)
        return torch.tensor(pred_vertex, dtype=torch.float32)#.to(device)
    
    def load_latentspace(self, save_path: str|None = None, 
                         file_name: str|None = None) -> np.ndarray|None:
        return self._load_npy('latentspaces', save_path, file_name)


class VertexTrainer24x6(VertexTrainer, BaseTrainer[Vertex24x6Config, AutoEncoderVertex24x6Dataset, 
                                                   VertexWrapper24x6]):
    def __init__(self, project_name: str, config_name: str | None = None, 
                 subconfig_name: str|None = None, load_from: str|None = None, config_dir: str = 'configs', 
                 config_kwargs: dict[str, Any] = {}):
        self.config_cls = Vertex24x6Config
        super().__init__(project_name, config_name, subconfig_name, load_from, config_dir, 
                         config_kwargs)
        self.dataset: AutoEncoderVertex24x6Dataset = self.dataset
        self.config: Vertex24x6Config = self.config

    def predict_3d(self, vertex_path: str, new_vertex: np.ndarray|None = None, 
                   load_from: Literal['best', 'last']|str|None = None, encode_only: bool = False) -> np.ndarray:
        pred = self.predict(vertex_path, new_vertex, load_from, encode_only)
        return self.dataset.to_3d_vertex(pred)
    
    def _predict_slice(self, vertex_path: str, fixed_idcs: list[int], other_k: int, dim: int, 
                       train_mode: TrainerModes|None = None, new_vertex: np.ndarray|None = None, 
                       load_from: Literal['best', 'last']|str|None = None, encode_only: bool = False) -> np.ndarray:
        assert all([i >= 1 and i <= self.dataset.n_freq for i in fixed_idcs]), \
            f"Any item in `fixed indices` must be in range [1,{self.dataset.n_freq}]."
        assert dim in [2, 4], "`dim` must be either 2 or 4."
        
        if dim == 2:
            assert len(fixed_idcs) == 4, "`fixed indices` must have length=4."
        elif dim == 4:
            assert len(fixed_idcs) == 2, "`fixed indices` must have length=2."
            assert other_k is not None and other_k >= 1 and other_k <= self.dataset.k_dim, \
                f"`other_k` must be in range [1,{self.dataset.k_dim}]"
            assert other_k != ((self.config.construction_axis + 1) // 2), \
                f"`other_k` must refer to a different k_i than `axis`."
        
        dataset: PredictVertex24x6Dataset = self.config.predict_dataset(self.config, vertex_path, new_vertex,
                                                                        dim=dim, fixed_idcs=fixed_idcs, 
                                                                        other_k=other_k)
        dataloader = self.data_loader(dataset, batch_size=dataset.length, num_workers=0, 
                                      persistent_workers=False)

        # predict
        if load_from:
            ckpt_path = self.get_model_ckpt(load_from)
            self.init_trainer(train_mode, Path(ckpt_path).parents[1])
        replace_at = dataset.replace_at
        pred_vertex = self.prepare_prediction_matrix(dim, encode_only, replace_at)
        self.wrapper.set_predictor(pred_vertex, encode_only, replace_at)
        self.trainer.predict(self.wrapper, dataloader, return_predictions=False, ckpt_path=ckpt_path)
        
        # save results to disk
        pred_vertex = self.wrapper.pred_vertex.cpu().numpy()
        if dim == 2:
            filename = f'{Path(vertex_path).stem}_' + '_'.join(map(str, fixed_idcs))
        elif dim == 4:
            filename = f'{Path(vertex_path).stem}_' + '_'.join(map(str, fixed_idcs)) + f'_k{other_k}'
        subfolder = 'latentspace_slices' if encode_only else 'prediction_slices'
        self.save_prediction(pred_vertex, filename, subfolder)
        return pred_vertex
    
    def predict_slice2d(self, vertex_path: str, fixed_idcs: list[int], train_mode: TrainerModes|None = None, 
                        new_vertex: np.ndarray|None = None, load_from: Literal['best', 'last']|str|None = None, 
                        encode_only: bool = False) -> np.ndarray:
        pred_vertex = self._predict_slice(vertex_path, fixed_idcs, None, 2, train_mode, new_vertex, 
                                          load_from, encode_only)
        return pred_vertex
    
    def predict_slice4d(self, vertex_path: str, fixed_idcs: list[int], other_k: int, 
                        train_mode: TrainerModes|None = None, new_vertex: np.ndarray|None = None, 
                        load_from: Literal['best', 'last']|str|None = None, encode_only: bool = False) -> np.ndarray:
        pred_vertex = self._predict_slice(vertex_path, fixed_idcs, other_k, 4, train_mode, new_vertex, 
                                          load_from, encode_only)
        return pred_vertex

    def load_latentspace_slice(self, save_path: str|None = None, 
                               file_name: str|None = None) -> np.ndarray|None:
        return self._load_npy('latentspace_slices', save_path, file_name)

    def load_prediction_slice(self, save_path: str|None = None, 
                              file_name: str|None = None) -> np.ndarray|None:
        return self._load_npy('prediction_slices', save_path, file_name)
