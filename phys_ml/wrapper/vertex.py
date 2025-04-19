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
            inputs = (batch[1], batch[0])
        else:
            inputs = batch[0]
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


class VertexWrapper24x6InfoNCE(VertexWrapper24x6):
    def __init__(self, config: Vertex24x6Config, in_dim: int):
        super().__init__(config, in_dim)
        self.nce = config.resolve_objectpath('info_nce.InfoNCE')(negative_mode='paired')
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flatten batch from shape(batch_size, 4, vector_length) to (batch_size * 4, vector_length)
        batch = (b.reshape((b.shape[0] * b.shape[1], b.shape[2])) for b in batch)
        if self.positional_encoding:
            inputs = (batch[0], batch[1])
        else:
            inputs = batch[0]
        return inputs, batch[2].float()
    
    def reshape_output(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = tensor.reshape((tensor.shape[0] // 4, 4, tensor.shape[1])).transpose(1, 0, 2)
        return tensor[0], tensor[1], tensor[2:].transpose(1, 0, 2)
    
    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = self.get_inputs_and_targets(batch)
        recons, latents = self.forward(inputs)
        latents_ref, latents_pos, latents_negs = self.reshape_output(latents)
        nce_loss = self.nce(latents_ref, latents_pos, latents_negs)
        rec_loss = self.criterion(recons, targets)
        loss = nce_loss + rec_loss
        return recons, targets, loss
