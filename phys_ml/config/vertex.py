from typing import TYPE_CHECKING

from . import Config


if TYPE_CHECKING:
    from .. import wrapper, models, load_data


class VertexConfig(Config['models.AutoEncoderVertex','wrapper.VertexWrapper', 'load_data.AutoEncoderVertexDataset']):
    construction_axis: int = 3
    sample_count_per_vertex: int = 2000
    positional_encoding: bool = True
    matrix_dim = 3
