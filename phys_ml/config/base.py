import importlib
import json
import pydoc

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Type, TypeVar, TYPE_CHECKING

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .. import wrapper, models, load_data
R = TypeVar('R', bound='models.BaseModule')
S = TypeVar('S', bound='wrapper.BaseWrapper')
T = TypeVar('T', bound='load_data.FilebasedDataset')


@dataclass
class Config(Generic[R, S, T]):
    _base_dir: str = Path(__file__).parent.parent.parent.as_posix()  # project base working directory
    model_name: str = 'BaseModule'                                   # name of the model class
    _model_wrapper: str = 'BaseWrapper'                              # name of the model wrapper class
    resume: Literal['best', 'last']|str|None = None                  # resume training from checkpoint from given path or
                                                                     #    'last' - most recent checkpoint, 
                                                                     #    'best' - checkpoint with best score
    save_dir: str = 'saves'                                          # base directory for saved outputs
    save_path: str = ''                                              # path to saved outputs
    path_train: str = ''                                             # path to training data
    _dataset: str = 'FilebasedDataset'                               # name of the dataset class
    _data_loader: str = 'torch.utils.data.DataLoader'                # name of the data loader class
    num_dataloader_workers: int = 8                                  # number of workers for the data loader
    sample_seed: int = 42                                            # seed for data sampling
    test_ratio: float = 0.2                                          # ratio of test data
    subset: int|float|None = None                                    # either number or ratio of data files to load, 
                                                                     #    if negative gives the number of files NOT to load
    subset_shuffle: bool = True                                      # randomly select data files
    subset_seed: int = 42                                            # seed for random data file selection
    
    # model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [])       # hidden dimensions of the model
    out_dim: int = 128                                               # output dimension of the model

    # training
    strategy: str|None = None                                        # training strategy for pytorch lightning trainer
    batch_size: int = 20                                             # batch size
    learning_rate: float = 0.0001                                    # learning rate
    weight_decay: float = 1e-05                                      # weight decay
    epochs: int = 1000                                               # number of epochs
    device_type: Literal['cpu', 'gpu', 'mps', 'xla', 'hpu'] = 'gpu'  # pytorch-lightning device type (accelerator)
    devices: int = 1                                                 # >= <# devices (CPUs or GPUs) on partition> * `num_nodes`
    num_nodes: int = 1                                               # number of nodes to use when training on the cluster

    # torch modules
    _criterion: str = 'torch.nn.MSELoss'                                  # name of the criterion class
    criterion_kwargs: dict[str, Any] = field(default_factory=lambda: {})  # keyword arguments for the criterion class
    _optimizer: str = 'torch.optim.AdamW'                                 # name of the optimizer class
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})  # keyword arguments for the optimizer class
    _activation = 'torch.nn.ReLU'                                         # name of the activation class
    activation_kwargs: dict[str, Any] = field(default_factory=lambda: {}) # keyword arguments for the activation class

    # lightning callbacks
    _model_checkpoint: str = 'ModelCheckpoint'                       # name of the model checkpoint class
    model_checkpoint_kwargs: dict[str, Any] = field(default_factory=lambda: {
        'save_top_k': 1,        # Save top 10 models
        'monitor': 'val_loss',  # Monitor validation loss
        'mode': 'min',          # 'min' for minimizing the validation loss
        'verbose': True,
        'save_last': True,
    })                                                               # keyword arguments for the model checkpoint class
    _callbacks: list[str] = field(default_factory=lambda: [])        # list of callback class names
    callbacks_kwargs: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        'EarlyStopping': {
            'monitor': 'val_loss',  # Monitor validation loss
            'mode': 'min',          # 'min' for minimizing the validation loss
            'patience': 10,         # Number of epochs with no improvement after which training will be stopped
            'verbose': True
        },
    })                                                               # dictionary of callback class names and their keyword arguments
    
    @classmethod
    def from_json(cls, config_name: str, subconfig_name: str|None = None, 
                  directory: str = 'configs', **kwargs) -> 'Config':
        """
        Creates a class containing all training parameters from a config-JSON. 
        Add kwargs to add attributes or overwrite attributes from the config-JSON.

        Parameters
        ----------
        config_name : str
            File name of the config-JSON.
        subconfig_name : str | None, optional
            Name of a sub-config in the config-file.\n
            Required if the file contains multiple cofigurations. (defaults to None)
        directory : str, optional
            Path to the directory of the config-file. (defaults to 'configs')
        **kwargs :
            Adds or overwrites attributes taken from the config-JSON.

        Returns
        -------
        Config
            A Config-instance with parameters set according to the JSON-file.
        """
        directory = Path(cls._base_dir, directory)
        with open(directory / config_name) as f:
            config: dict[str, Any] = json.load(f)
        if subconfig_name is not None:
            config = config[subconfig_name]
        config.update(kwargs)
        conf = cls()
        for key, value in config.items():
            setattr(conf, key.lower(), value)
        return conf

    @property
    def base_dir(self) -> Path:
        return Path(self._base_dir)
    
    @base_dir.setter
    def base_dir(self, value: str):
        self._base_dir = value
    
    @property
    def model(self) -> Type[R]:
        return self.resolve_objectname(self.model_name, '..models')
    
    @model.setter
    def model(self, value: str):
        self.model_name = value

    @property
    def model_wrapper(self) -> Type[S]:
        return self.resolve_objectname(self._model_wrapper, '..wrapper')
    
    @model_wrapper.setter
    def model_wrapper(self, value: str):
        self._model_wrapper = value
    
    @property
    def dataset(self) -> Type[T]:
        return self.resolve_objectname(self._dataset, '..load_data')
    
    @dataset.setter
    def dataset(self, value: str):
        self._dataset = value
    
    @property
    def data_loader(self) -> type[DataLoader]:
        if '.' in self._data_loader:
            return self.resolve_objectpath(self._data_loader)
        else:
            return self.resolve_objectname(self._data_loader, '..load_data')
    
    @data_loader.setter
    def data_loader(self, value: str):
        self._data_loader = value
    
    @property
    def criterion(self) -> type[Module]:
        return self.resolve_objectpath(self._criterion)
    
    @criterion.setter
    def criterion(self, value: str):
        self._criterion = value
    
    def get_criterion(self, **kwargs) -> Module:
        """Get an instantiation of the criterion class-object using `self.criterion_kwargs` as arguments. 
        Use `**kwargs` to overwrite arguments set in `self.criterion_kwargs`."""
        return self._instantiate_type('criterion', **kwargs)
    
    @property
    def optimizer(self) -> type[Optimizer]:
        return self.resolve_objectpath(self._optimizer)
    
    @optimizer.setter
    def optimizer(self, value: str):
        self._optimizer = value
    
    def get_optimizer(self, **kwargs) -> Optimizer:
        """Get an instantiation of the optimizer class-object using `self.optimizer_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.optimizer_kwargs`."""
        return self._instantiate_type('optimizer', **kwargs)
    
    @property
    def activation(self) -> type[Module]:
        return self.resolve_objectpath(self._activation)
    
    @activation.setter
    def activation(self, value: str):
        self._activation = value
    
    def get_activation(self, **kwargs) -> Module:
        """Get an instantiation of the activation class-object using `self.activation_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.activation_kwargs`."""
        return self._instantiate_type('activation', **kwargs)
    
    @property
    def model_checkpoint(self) -> type[ModelCheckpoint]:
        return self.resolve_objectname(self._model_checkpoint, 'lightning.pytorch.callbacks')
    
    @model_checkpoint.setter
    def model_checkpoint(self, value: str):
        self._model_checkpoint = value
    
    def get_model_checkpoint(self, **kwargs) -> ModelCheckpoint:
        """Get an instantiation of the model_checkpoint class-object using `self.model_checkpoint_kwargs` as arguments.
        Use `**kwargs` to overwrite arguments set in `self.model_checkpoint_kwargs`."""
        return self._instantiate_type('model_checkpoint', **kwargs)
    
    @property
    def callbacks(self) -> list[type[Callback]]:
        if not self._callbacks:
            return []
        return [self.resolve_objectname(cb, 'lightning.pytorch.callbacks') for cb in self._callbacks]
    
    @callbacks.setter
    def callbacks(self, value: list[str]):
        self._callbacks = value
    
    def get_callbacks(self, kwargs: dict[str, dict[str, Any]] = {}) -> list[Callback]:
        """Get a list of instantiations of the callback class-objects using `self.callbacks_kwargs` as arguments.
        Use `kwargs` to overwrite arguments set in `self.callbacks_kwargs`. 
        Attention: The `kwargs`-argument is a dictionary of kwarg-dictionaries using the callback-class-names as keys."""
        return [cb(**dict(self.callbacks_kwargs[cb.__name__], **kwargs.get(cb.__name__, {}))) for cb in self.callbacks]
    
    @staticmethod
    def resolve_objectpath(obj_fqn: str) -> type:
        """
        Import a class-object defined by its fully qualified name.

        Parameters
        ----------
        obj_fqn : str
            The fully qualified name of the class.

        Returns
        -------
        type
            The class.
        """
        return pydoc.locate(obj_fqn)
    
    @staticmethod
    def resolve_objectname(obj_name: str, module_name: str) -> type:
        """
        Import a class object defined by the class-name and its fully qualified module-name.

        Parameters
        ----------
        obj_name : str
            The name of the class.
        module_name : str
            The name of the class.

        Returns
        -------
        type
            The class.
        """
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, obj_name)
    
    def _instantiate_type(self, name: str, **kwargs) -> Any:
        """Instantiate a class-object using the kwargs from the corresponding class-attribute. 
        Kwargs from the class-attribute can be overwritten by providing kwrags in the method-call."""
        attr_kwargs = dict(getattr(self, name + '_kwargs'), **kwargs)
        return getattr(self, name)(**attr_kwargs)
    
    def __repr__(self):
        return self.__class__.__qualname__ + '(\n   ' + ',\n   '.join([f"{k}={v!r}" for k, v in self.__dict__.items()]) + '\n)'

    def __str__(self):
        return self.__class__.__qualname__ + ':\n{\n   ' + ',\n   '.join([f"{k}: {v!r}" for k, v in self.__dict__.items()]) + '\n}'
    
    def __iter__(self):
        return iter(self.__dict__.items())

    def __len__(self):
        return len(self.__dict__)

    def as_dict(self):
        return self.__dict__
    
    def save(self, path: str) -> None:
        """Save config as JSON file."""
        save_dict = {self.model_name.upper(): self.as_dict()}
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=4)
