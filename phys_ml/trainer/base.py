import datetime
import glob
import json
import logging
import os

from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment

import torch
from torch.utils.data import DataLoader, random_split

from .. import config
from ..load_data import FilebasedDataset, SimpleDataset
from ..wrapper import BaseWrapper


R = TypeVar('R', bound=BaseWrapper)
S = TypeVar('S', bound=FilebasedDataset)
T = TypeVar('T', bound=config.Config)


class TrainerModes(IntEnum):
    SLURM = 1
    JUPYTER = 2


class CKPT_TYPE(StrEnum):
    BEST = 'best'
    LAST = 'last'


class BaseTrainer(Generic[T, S, R]):
    config_cls: type[T] = T.__bound__

    def __init__(self, project_name: str, config_name: str | None = None, subconfig_name: str|None = None, 
                 load_from: str|None = None, dataset: S|None = None, config_dir: str = 'configs', config_kwargs: dict[str, Any] = {}):
        """
        Main class for training and using a model.

        Parameters
        ----------
        project_name : str
            Some project name under which the results are saved.
        train_mode : TrainerModes
            TrainerModes configure the lightning-trainer for different environments 
            (e.g. cluster, local, jupyter).
        config_name : str
            Name of the config-file to load.
            Required if `load_from` not provided. (defaults to None)
        subconfig_name : str | None, optional
            Name of a subsection in the config-file.\n
            Required if the file contains multiple config-sections. (defaults to None)
        load_from : str | None, optional
            Loads config and optionally checkpoints if `config.resume` is enabled from other training-run.\n
            Absolute path should end as `<saves_dir>/<project_name>/<version>/`.\n
            Path can also be relative to `/<project_name>/`. (defaults to None)
        dataset : FilebasedDataset | None, optional
            Use existing / pre-loaded dataset for training.\n
        config_dir : str, optional
            Path to the directory of the config-files. (defaults to `'configs'`)
        config_kwargs : dict[str, Any], optional
            Additional kwargs that are applied to loading the config-file.\n
            Allows to overwrite attributes from the config-file. (defaults to {})
        """
        conf_classname = repr(self.__orig_bases__[-1]).split('[')[1].split(',')[0].split('.')[-1]
        self.config_cls = getattr(config, conf_classname, config.Config)
        self.config: T = None

        self.project_name = project_name
        self.subconfig_name = subconfig_name
        self.load_from = load_from

        if load_from:
            self.load_from = self.get_full_save_path(load_from)
            self.load_config_from_saves(self.load_from, **config_kwargs)
        else:
            self.config: T = self.config_cls.from_json(config_name, subconfig_name, config_dir, **config_kwargs)
        self.data_loader: type[DataLoader] = self.config.data_loader
        self.dataset = dataset
        self.console_logger = logging.getLogger('lightning.pytorch')
        self.wrapper: R = None
        self.trainer: Trainer = None
    
    @property
    def input_size(self) -> int|np.ndarray:
        """
        Get the input size of the model. 
        Overwrite when having a different dataset item structure.
        """
        return self.dataset[0][0].shape[0]
    
    @property
    def save_prefix(self) -> str:
        """
        Get the name of the saves-subfolder for this training-run.\n
        Overwrite for custom naming.
        """
        return f"save_{self.subconfig_name}_BS{self.config.batch_size}_"

    def train(self, train_mode: TrainerModes, 
              load_ckpt: Literal[CKPT_TYPE.BEST, CKPT_TYPE.LAST]|str|None = None) -> None:
        """
        Main method to train the model.

        Parameters
        ----------
        load_ckpt : Literal['last', 'best'] | str | None, optional
            Load model from the last or best checkpoint or from a specific path before training. (defaults to None)
        """
        if self.dataset is None:
            self.dataset = self.config.dataset(self.config)
        load_ckpt = load_ckpt or self.config.resume
        save_path = self.get_full_save_path()
        self.config.save_path = save_path.as_posix()
        ckpt_path = self.init_trainer(train_mode, load_ckpt)

        ''' Saving config-file '''
        json_object = json.dumps(self.config.as_dict(), indent=4)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'config.json', 'w') as outfile:
            outfile.write(json_object)
        
        self.pre_train()
        self._train(ckpt_path)
        self.post_train()
    
    def _train(self, ckpt_path: Path|None = None) -> None:
        """
        Core method for running the training.\n
        Overwrite for custom training process.

        Parameters
        ----------
        ckpt_path : str | None, optional
            Path to checkpoint to resume training from. (defaults to None)
        """
        ''' Dataloading '''
        train_dataloader, validation_dataloader = self.create_data_loader()
        
        ''' Train '''
        self.trainer.fit(self.wrapper, train_dataloader, validation_dataloader, ckpt_path=ckpt_path)
    
    def pre_train(self) -> None:
        """
        Overwrite to perform operations before starting the training.
        """
        pass

    def post_train(self) -> None:
        """
        Overwrite to perform operations after finishing training.
        """
        pass

    def predict(self, new_data_path: str, new_data: np.ndarray|None = None, 
                load_ckpt: Literal[CKPT_TYPE.BEST, CKPT_TYPE.LAST]|str|None = None) -> np.ndarray:
        """
        Performs a prediction using new data.

        Parameters
        ----------
        new_data_path : str
            Path to file containing the new data to predict on.
        new_data : np.ndarray | None, optional
            New data to predict on. 
            If `None` load data from `new_data_path`. (defaults to None)
        load_ckpt : Literal['last', 'best'] | str | None, optional
            Load model from the last or best checkpoint or from a specific path. 
            If `None` uses an already loaded model. (defaults to None)

        Returns
        -------
        np.ndarray
            Prediction as numpy array.
        """
        if new_data is None:
            new_data = self.dataset.load_from_file(new_data_path)
        if load_ckpt:
            _ = self.load_model(self.load_from, load_ckpt, predict=True)
        dataset = SimpleDataset(new_data)
        dataloader = self.data_loader(dataset, batch_size=self.config.batch_size, 
                                      num_workers=self.config.num_dataloader_workers, 
                                      persistent_workers=bool(self.config.num_dataloader_workers),
                                      pin_memory=True)
        pred = self.trainer.predict(self.wrapper, dataloader, ckpt_path=load_ckpt)
        pred = torch.concat(pred, dim=0)
        pred = pred.cpu().numpy()
        self.save_prediction(pred, Path(new_data_path).stem, 'predictions')
        return pred
    
    def init_trainer(self, train_mode: TrainerModes, 
                     load_ckpt: Literal[CKPT_TYPE.BEST, CKPT_TYPE.LAST]|str|None) -> Path:
        ckpt_path = self.load_model(self.load_from, load_ckpt, predict=False)
        
        ''' Logging and checkpoint saving '''
        logger = self.set_logging()

        ''' Set pytorch_lightning Trainer '''
        callbacks = [self.config.get_model_checkpoint(), *self.config.get_callbacks()]
        trainer_kwargs = {'max_epochs': self.config.epochs, 'accelerator': self.config.device_type, 
                          'devices': self.config.devices, 'logger': logger, 'callbacks': callbacks, 
                          'log_every_n_steps': self.config.log_every_n_steps}
        if train_mode == TrainerModes.SLURM:
            strategy = self.config.strategy or 'ddp'
            self.trainer = Trainer(num_nodes=self.config.num_nodes, strategy=strategy, **trainer_kwargs)
        elif train_mode == TrainerModes.JUPYTER:
            strategy = self.config.strategy or ('ddp_notebook' if os.name == 'posix' else 'auto')
            self.trainer = Trainer(strategy=strategy, plugins=[LightningEnvironment()], **trainer_kwargs)
        return ckpt_path
    
    def save_prediction(self, vertex_prediction: np.ndarray, filename: str, subfolder: str):
        p = self.get_full_save_path() / subfolder
        p.mkdir(exist_ok=True)
        fp = p / f'{filename}.npy'
        np.save(fp, vertex_prediction)
    
    def set_logging(self) -> TensorBoardLogger:
        """
        Set TensorBoardLogger and ModelCheckpoint.\n
        Overwrite for custom logging.
        """
        save_path = Path(self.config.save_path)
        run_path = save_path.parent.name
        version = save_path.name
        logger = TensorBoardLogger(save_path.parent.parent, name=run_path, version=version)
        return logger

    def create_data_loader(self) -> tuple[DataLoader, DataLoader]:
        """
        Create DataLoader.\n
        Overwrite for custom data loading.
        """
        train_set, validation_set = random_split(self.dataset, [1 - self.config.test_ratio, self.config.test_ratio], 
                                                 generator=torch.Generator().manual_seed(42))
        train_dataloader = self.data_loader(train_set, batch_size=self.config.batch_size, shuffle=True, 
                                            num_workers=self.config.num_dataloader_workers, 
                                            persistent_workers=bool(self.config.num_dataloader_workers), 
                                            pin_memory=True)
        validation_dataloader = self.data_loader(validation_set, batch_size=self.config.batch_size, 
                                                 num_workers=self.config.num_dataloader_workers, 
                                                 persistent_workers=bool(self.config.num_dataloader_workers), 
                                                 pin_memory=True)
        return train_dataloader, validation_dataloader
    
    def get_full_save_path(self, save_path: str|None = None) -> Path:
        """
        Return the absolute save-path from a path relative to `<saves_dir>/<project_name>/`.

        Parameters
        ----------
        save_path : str | None, optional
            Path relative to `<saves_dir>/<project_name>/`.
            If `None` `self.config.save_path` is used. (defaults to None)

        Returns
        -------
        Path
            The absolute save-path.
        """
        if save_path is None:
            save_path = self.config.save_path if self.config else ''
        save_path = Path(save_path)
        if save_path.is_absolute():
            return save_path.resolve()
        
        if self.config is None:
            run_path = self.config_cls.save_dir or self.save_prefix + str(datetime.datetime.now().date())
            pref = Path(self.config_cls._base_dir) / run_path
        else:
            run_path = self.config.save_dir or self.save_prefix + str(datetime.datetime.now().date())
            pref = self.config.base_dir / run_path
        return (pref / self.project_name / save_path).resolve()
    
    def get_model_ckpt(self, load_from: str|None, load_ckpt: Literal[CKPT_TYPE.BEST, CKPT_TYPE.LAST]|str|None) -> Path|None:
        """
        Gets the path to the best or latest model checkpoint or to the most recent checkpoint in a given directory.
        If trainer not fitted, gets the checkpoint with the highest epoch from the saved checkpoints.
        Always returns the best checkpoint if `Modelchekpoint.save_top_k=1`.

        Parameters
        ----------
        load_from : str | None
            Absolute save-path to a different model-save-folder (like `/<project_name>/<version>/`).\n
            If `None` use the current save-path.
        load_ckpt : Literal['best', 'last'] | str | None
            Load model from the last or best checkpoint or from a specific path.
            If `None` return load the newest checkpoint.

        Returns
        -------
        str
            Path to the model checkpoint.
        """
        ckpt_path = None
        if load_ckpt is not None and load_ckpt.endswith('.ckpt'):
            ckpt_path = load_ckpt
        else:
            CKPT_DIRNAME = 'checkpoints'
            if load_ckpt is None or load_ckpt == CKPT_TYPE.BEST or load_ckpt == CKPT_TYPE.LAST:
                ckpt_dir = Path(load_from or self.config.save_path) / CKPT_DIRNAME
            elif CKPT_DIRNAME not in load_ckpt:
                ckpt_dir = Path(load_ckpt) / CKPT_DIRNAME
            
            # get 'last'-checkpoint
            if self.trainer:
                last_name = self.trainer.checkpoint_callback.CHECKPOINT_NAME_LAST
            else:
                last_name = self.config.get_model_checkpoint().CHECKPOINT_NAME_LAST
            last_name = f'{last_name}.ckpt'
            last_ckpt_path = ckpt_dir / last_name
            if not last_ckpt_path.exists():
                last_ckpt_path = None
            elif load_ckpt == CKPT_TYPE.LAST:
                ckpt_path = last_ckpt_path
            
            # get most recent or 'last' checkpoint
            if ckpt_path is None:
                ckpt_paths = (ckpt_dir).glob('*.ckpt')
                ckpt_paths = sorted(ckpt_paths, key=os.path.getctime)
                for path in reversed(ckpt_paths):
                    if path.name != last_name:
                        ckpt_path = path
                        break
            if ckpt_path is None and last_ckpt_path is not None:
                ckpt_path = last_ckpt_path
        if ckpt_path is None:
            self.console_logger.log(logging.INFO, 'Loading checkpoint: No checkpoint found.')
        elif not ckpt_path.exists():
            self.console_logger.log(logging.INFO, f'Loading checkpoint: No file found at {ckpt_path}.')
            ckpt_path = None
        return ckpt_path
    
    def load_model(self, load_from: Path|None, load_ckpt: Literal[CKPT_TYPE.BEST, CKPT_TYPE.LAST]|str|None, 
                   predict: bool = False, encode_only: bool = False) -> Path:
        ckpt_path = self.get_model_ckpt(load_from, load_ckpt)
        if ckpt_path:
            self.console_logger.log(logging.INFO, f"Loading model from checkpoint: {ckpt_path}")
            self.wrapper = self.config.model_wrapper.load_from_checkpoint(ckpt_path, config=self.config, 
                                                                            in_dim=self.input_size)
        if self.wrapper is None:
            self.wrapper = self.config.model_wrapper(self.config, self.input_size)
        self.wrapper.encode_only = encode_only
        if predict:
            self.wrapper.model.eval()
        return ckpt_path

    def load_config_from_saves(self, load_from: str, **kwargs) -> None:
        """
        Sets the used Config from a config-JSON file.

        Parameters
        ----------
        load_from : str
            File-path to load the config-JSON from.
        **kwargs :
            Additional keyword arguments that overwrite attributes in the config-class.
        """
        self.config = self.config_cls.from_json('config.json', directory=load_from, **kwargs)

    def _load_npy(self, npy_type: str, save_path: str|None = None, 
                  file_name: str|None = None) -> np.ndarray|None:
        """
        Load result arrays stored in the numpy format (`.npy`).

        Parameters
        ----------
        npy_type : str
            Name of subfolder in the model-save-folder where the numpy-files are stored.
        save_path : str | None, optional
            Path relative to `<saves_dir>/<project_name>/`.
            Required if no model loaded.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        file_name : str | None, optional
            Name of the `.npy`-file.\n
            If `None` the newest file in the folder is loaded. (defaults to None)

        Returns
        -------
        np.ndarray|None
            The loaded numpy array or `None` if the file does not exist.
        """
        if save_path:
            self.load_config_from_saves(self.get_full_save_path(save_path))
        p = self.get_full_save_path() / npy_type
        if not file_name:
            npy_path = max(p.glob(f'*.npy'), key=os.path.getctime)
        else:
            npy_path = p / file_name
        if os.path.exists(npy_path):
            return np.load(npy_path)
    
    def load_prediction(self, save_path: str|None = None, file_name: str|None = None) -> np.ndarray|None:
        """
        Load a previous prediction stored in the numpy format (`.npy`).

        Parameters
        ----------
        save_path : str | None, optional
            Path to the model-save-folder (like `/<project_name>/<version>/`).\n
            Required if no model loaded.\n
            If `None` `self.config.save_path` is used. (defaults to None)
        file_name : str | None, optional
            Name of the `.npy`-file.\n
            If `None` the newest file in the folder is loaded. (defaults to None)

        Returns
        -------
        np.ndarray|None
            The loaded numpy array or `None` if the file does not exist.
        """
        return self._load_npy('predictions', save_path, file_name)
    
    @staticmethod
    def get_device_from_accelerator(accelerator: str) -> str:
        """
        Maps lightning accelerators to pytorch device-types.

        Parameters
        ----------
        accelerator : str
            Lightning accelerator type name.

        Returns
        -------
        str
            Pytorch device type name.

        Raises
        ------
        ValueError
            If the accelerator is not known.
        """
        mapping = {
            "cpu": "cpu",
            "gpu": "cuda",
            "mps": "mps",
            "xla": "tpu",
            "hpu": "hpu"
        }
        if accelerator in mapping:
            return mapping[accelerator]
        else:
            raise ValueError(f"Unable to map accelerator: {accelerator}")
