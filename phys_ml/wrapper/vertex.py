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
        ndims = len(idcs)
        if isinstance(idcs, list):
            idcs = torch.stack(idcs).T
        if self.encode_only:
            pred = self.model.encode(inputs)
            if self.pred_vertex is not None:
                self.pred_vertex[*[idcs[:, i] for i in range(self.replace_at)], :, 
                                 *[idcs[:, i] for i in range(self.replace_at + 1, ndims)]] = pred
            return pred
        else:
            pred = self.model(inputs)
            if self.pred_vertex is not None:
                self.pred_vertex[*[idcs[:, i] for i in range(self.replace_at)], :, 
                                 *[idcs[:, i] for i in range(self.replace_at + 1, ndims)]] = pred
            return pred


class VertexWrapper24x6(VertexWrapper):
    def __init__(self, config: Vertex24x6Config, in_dim: int):
        super().__init__(config, in_dim)


class VertexWrapper24x6InfoNCE(VertexWrapper24x6):
    def __init__(self, config: Vertex24x6Config, in_dim: int):
        super().__init__(config, in_dim)
        self.n_ct_samples = len(self.config.subset_type) + 1 if self.config.subset_type else 4
        self.nce = config.resolve_objectpath('info_nce.InfoNCE')(negative_mode='paired')
    
    def get_inputs_and_targets(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_vectors, idcs, targets = batch
        samples = input_vectors.reshape((-1, input_vectors.shape[-1]))
        if self.positional_encoding:
            inputs = (samples, idcs.reshape((-1, idcs.shape[-1])))
        else:
            inputs = samples
        return inputs, targets.reshape((-1, targets.shape[-1])).float()
    
    def reshape_output(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor = tensor.reshape((tensor.shape[0] // self.n_ct_samples, self.n_ct_samples, tensor.shape[1])).transpose(1, 0)
        return tensor[0], tensor[1], tensor[2:].transpose(1, 0)
    
    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = self.get_inputs_and_targets(batch)
        recons, latents = self.forward(inputs)
        latents_ref, latents_pos, latents_negs = self.reshape_output(latents)
        nce_loss = self.nce(latents_ref, latents_pos, latents_negs)
        rec_loss = self.criterion(recons, targets)
        loss = nce_loss + rec_loss
        return recons, targets, loss
    
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        inputs, idcs = batch
        ndims = len(idcs)
        if isinstance(idcs, list):
            idcs = torch.stack(idcs).T
        if self.encode_only:
            pred = self.model.encode(inputs)
            if self.pred_vertex is not None:
                self.pred_vertex[*[idcs[:, i] for i in range(self.replace_at)], :, 
                                 *[idcs[:, i] for i in range(self.replace_at + 1, ndims)]] = pred
            return pred
        else:
            pred, latent = self.model(inputs)
            if self.pred_vertex is not None:
                self.pred_vertex[*[idcs[:, i] for i in range(self.replace_at)], :, 
                                 *[idcs[:, i] for i in range(self.replace_at + 1, ndims)]] = pred
            return pred
