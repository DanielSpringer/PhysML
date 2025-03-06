import torch

from ..config.vertex import *
from ..models.vertex import *
from . import BaseWrapper


class VertexWrapper(BaseWrapper[AutoEncoderVertex, VertexConfig]):
    ''' Wrapper for the vertex compression '''
    def __init__(self, config: VertexConfig, in_dim: int):
        super().__init__(config, in_dim)
        self.positional_encoding = config.positional_encoding
        
        # needed for prediciton only
        self.encode_only: bool = None
        self.pred_vertex: torch.Tensor = None
        self.replace_at: int = None
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positional_encoding:
            inputs = (batch[0], batch[1])
        else:
            inputs = batch[1]
        return inputs, batch[2].float()

    def set_predictor(self, pred_vertex: torch.Tensor, encode_only: bool = False, replace_at: int|None = None):
        self.encode_only = encode_only
        self.pred_vertex: torch.Tensor = pred_vertex
        self.replace_at = replace_at if replace_at is not None else self.config.construction_axis - 1
    
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        inputs, idcs = batch
        if self.encode_only:
            pred = self.model.encode(inputs)
            return pred
        else:
            pred = self.model(inputs)
            for i, p in enumerate(pred):
                idx = [idx[i] for idx in idcs]
                idx[self.replace_at] = slice(None)
                self.pred_vertex[*idx] = p


class VertexWrapper24x6(VertexWrapper):
    def __init__(self, config: Vertex24x6Config, in_dim: int):
        super().__init__(config, in_dim)


class VertexWrapper24x6Sparse(VertexWrapper24x6):
    def __init__(self, config: Vertex24x6SparseConfig, in_dim: int):
        super().__init__(config, in_dim)
    
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        inputs, idcs = batch
        if self.encode_only:
            pred = self.model.encode(inputs)
            return pred
        else:
            pred = self.model(inputs)
            for i, p in enumerate(pred):
                full_vector = self.config.dataset.sparse_to_full_vector(p)
                idx = [idx[i] for idx in idcs]
                idx[self.replace_at] = slice(None)
                self.pred_vertex[*idx] = full_vector
