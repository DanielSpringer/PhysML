from typing import TYPE_CHECKING

from . import Config


if TYPE_CHECKING:
    from .. import wrapper, models, load_data


class VertexConfig(Config['models.AutoEncoderVertex','wrapper.VertexWrapper', 
                          'load_data.AutoEncoderVertexDataset']):
    construction_axis: int = 3
    sample_count_per_vertex: int = 2000
    positional_encoding: bool = False
    matrix_dim = 3

    _predict_dataset: str = 'PredictVertexDataset'

    @property
    def predict_dataset(self) -> type['load_data.vertex.PredictVertexDataset']:
        return self.resolve_objectname(self._predict_dataset, '..load_data')
    
    @predict_dataset.setter
    def predict_dataset(self, value: str):
        self._predict_dataset = value


class Vertex24x6Config(VertexConfig, Config['models.AutoEncoderVertex','wrapper.VertexWrapper24x6', 
                                            'load_data.AutoEncoderVertex24x6Dataset']):
    sample_count_per_vertex: int = 2000
    matrix_dim = 6

    _predict_dataset: str = 'PredictVertex24x6Dataset'
