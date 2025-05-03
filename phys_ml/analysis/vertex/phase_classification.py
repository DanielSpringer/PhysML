import os
import pickle

from pathlib import Path
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.environments import LightningEnvironment
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from phys_ml.load_data.base import SimpleDataset
from phys_ml.load_data.vertex import AutoEncoderVertex24x6Dataset
from phys_ml.trainer.vertex import VertexTrainer24x6


class PhaseClassification:
    def __init__(self, model_path: str, models: list[BaseEstimator], version: int, 
                 mode: Literal['c', 'r'] = 'c', latent_space_paths: list[str]|None = None):
        self.mode = mode
        self.models = models
        self.version = version
        self.encode = model_path is not None
        self.file_type: Literal['vertex', 'latent_space'] = None
        if not latent_space_paths:
            self.file_type = 'vertex'
        else:
            self.file_type = 'latent_space'
        if self.encode:
            self.vertex_trainer = VertexTrainer24x6(project_name='vertex_24x6', load_from=model_path)
            _ = self.vertex_trainer.load_model(load_from=model_path, predict=True, encode_only=True)
            self.dataset = self.vertex_trainer.dataset
            self.ls_length = self.vertex_trainer.config.hidden_dims[-1]
            self.device = self.vertex_trainer.get_device_from_accelerator(self.vertex_trainer.config.device_type)
        else:
            self.dataset = AutoEncoderVertex24x6Dataset
            self.ls_length = self.dataset.length * self.dataset.dim
        self.load_models()
        self.predict_samples: tuple[np.ndarray, list[int]] = None
    
    def get_phase_from_filepath(self, file_path: str) -> float|int:
        fname = Path(file_path).stem
        tp, mu = (float(s[2:]) for s in fname.split('_'))
        if self.mode == 'r':
            return tp
        else:
            for i, (phase, borders) in enumerate(AutoEncoderVertex24x6Dataset.phase_borders.items()):
                if tp >= borders[0] and tp < borders[1]:
                    return i
    
    def predict_latent_space_vectors(self, vertex: np.ndarray, samples_per_vertex: int) -> np.ndarray:
        # NOTE: also possible with dataloader and trainer.predict
        input_vectors, input_idcs = self.dataset.sample(vertex, samples_per_vertex)
        if self.encode:
            input_vectors = torch.tensor(input_vectors, dtype=torch.float32).to(self.device)
            ls_vectors = self.vertex_trainer.wrapper.predict_step((input_vectors, input_idcs)).detach().cpu().numpy()
        else:
            ls_vectors = np.array(input_vectors)
        return ls_vectors

    def load_data(self, file_paths: list[str], samples_per_vertex: int) -> tuple[np.ndarray, list[int|float]]:
        assert len(file_paths) > 0, 'List `file_paths` is empty.'
        inputs = np.empty((0, self.ls_length))
        targets = []
        for fp in tqdm(np.random.permutation(file_paths), desc='Load data'):
            phase = self.get_phase_from_filepath(fp)
            if self.file_type == 'vertex':
                vertex = self.dataset.load_from_file(fp)
                ls_vectors = self.predict_latent_space_vectors(vertex, samples_per_vertex)
            else:
                ls_vectors = np.load(fp)
            inputs = np.concatenate((inputs, ls_vectors), axis=0)
            targets.extend([phase] * samples_per_vertex)
        return inputs, targets

    def train(self, train_files: list[str], samples_per_vertex: int) -> dict[str, np.ndarray]:
        inputs, targets = self.load_data(train_files, samples_per_vertex)
        for model in tqdm(self.models, desc='Fit models'):
            try:
                model.fit(inputs, targets)
                with open(f'{self.version:02}_{repr(model)}.pkl','wb') as f:
                    pickle.dump(model,f)
            except Exception as e:
                print(f'Could not fit model {repr(model)}:\n\t{e}')

    def load_models(self):
        for i, model in enumerate(self.models):
            try:
                check_is_fitted(model)
            except:
                try:
                    with open(f'{self.version:02}_{repr(model)}.pkl', 'rb') as f:
                        self.models[i] = pickle.load(f)
                except:
                    print(f'Model {repr(model)} is not fitted and could not be loaded. '
                          'Call train() before using the PhaseClassifier.')

    def evaluate_model(self, model, test_ls_vectors: np.ndarray, test_targets: list[int|float], labels: list[str],
                       print_conf_mat: bool = True, 
                       normalize: Literal['true', 'pred', 'all'] = 'true') -> tuple[dict[str, float], np.ndarray]:
        try:
            pred: np.ndarray = model.predict(test_ls_vectors)
            scores = {}
            if self.mode == 'r':
                scores['rmse'] = metrics.root_mean_squared_error(test_targets, pred)
                pmin, pmax = 0.0, 0.5
                pred = np.clip(pred, pmin, pmax)
                pred = [k for p in pred for k, v in AutoEncoderVertex24x6Dataset.phase_borders.items() 
                        if p >= v[0] and p < v[1]]
                test_targets = [k for t in test_targets for k, v in AutoEncoderVertex24x6Dataset.phase_borders.items()
                                if t >= v[0] and t < v[1]]
            scores.update({
                'acc': metrics.balanced_accuracy_score(test_targets, pred),
                'prec': metrics.precision_score(test_targets, pred, average='micro'),
                'rec': metrics.recall_score(test_targets, pred, average='micro'),
                'f1': metrics.f1_score(test_targets, pred, average='micro'),
            })
            if self.mode == 'c':
                labels = np.array(labels)
                pred = labels[pred]
                test_targets = labels[test_targets]
            conf_mat = metrics.confusion_matrix(test_targets, pred, labels=labels, normalize=normalize)
            if print_conf_mat:
                sns.set_theme(font_scale=1.4)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax = sns.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels, 
                                vmin=0.0, vmax=1.0, fmt=".2f", ax=ax, square=True, annot_kws={"size": 14})
                ax.tick_params(left=False, bottom=False)
                plt.xlabel('predicted')
                plt.ylabel('true')
                plt.title(repr(model))
                plt.show()
            return scores, conf_mat
        except Exception as e:
            print(e)
            print(f'Skipping model {repr(model)}, which could not be fitted.')
            return None, None

    def evaluate_classifiers(self, test_files: list[str], samples_per_vertex: int, print_conf_mat: bool = True,
                             normalize: Literal['true', 'pred', 'all'] = 'true') \
                                -> dict[str, tuple[dict[str, float], np.ndarray]]:
        self.load_models()
        results = {}
        inputs, targets = self.load_data(test_files, samples_per_vertex)
        labels = list(AutoEncoderVertex24x6Dataset.phase_borders.keys())
        for model in tqdm(self.models, desc='Predict'):
            scores, conf_mat = self.evaluate_model(model, inputs, targets, labels, print_conf_mat, normalize)
            results[repr(model)] = (scores, conf_mat)
        return results


class PolynomialRegression(LinearRegression):
    def __init__(self, fit_intercept: bool = True, copy_X: bool = True, n_jobs: int|None = -1, positive: bool = True):
        self.pf: PolynomialFeatures = None
        self.params: dict[str, Any] = None
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
    
    @classmethod
    def new(cls, degree: int|tuple[int, int] = 2, interaction_only: bool = False, 
            include_bias: bool = False, order: Literal['C', 'F'] = 'C', fit_intercept: bool = True, 
            copy_X: bool = True, n_jobs: int|None = -1, positive: bool = True) -> 'PolynomialRegression':
        # used for __repr__:
        defaults = {k: v for k, v in zip(list(cls.new.__annotations__.keys())[:-1], cls.new.__defaults__)}
        params = {}
        for attr, value in defaults.items():
            if (param := locals()[attr]) != value:
                params[attr] = param
        pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
        model = cls(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        model.pf = pf
        model.params = params
        return model
    
    def fit(self, X: np.ndarray, y, sample_weight = None):
        poly_features = self.pf.fit_transform(X)
        return super().fit(poly_features, y, sample_weight)
    
    def predict(self, X: np.ndarray):
        poly_features = self.pf.transform(X)
        return super().predict(poly_features)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={v}" for k, v in self.params.items()])})'


class NeuralNetClassifier:
    class NeuralNet(torch.nn.Module):
        def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int = 3):
            super().__init__()
            activation = nn.ReLU()
            layers = [nn.Linear(in_dim, hidden_dims[0]), activation]
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
            layers.append(nn.Sigmoid())
            self.layers = nn.Sequential(*layers)

        def forward(self, data_in) -> torch.Tensor:
            return self.layers(data_in)
    
    class Wrapper(LightningModule):
        def __init__(self, criterion: torch.nn, optimizer: type[torch.optim.Optimizer], 
                     in_dim: int, hidden_dims: list[int], out_dim: int = 3, learning_rate: float = 1e-3, 
                     weight_decay: float = 0.0):
            super().__init__()
            self.criterion = criterion
            self.optimizer = optimizer
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.model = NeuralNetClassifier.NeuralNet(in_dim, hidden_dims, out_dim)
        
        def __call__(self, *args, **kwds) -> torch.Tensor:
            return super().__call__(*args, **kwds)
        
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return self.model(batch)
        
        def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            inputs, targets = batch
            pred = self.forward(inputs)
            loss = self.criterion(pred, targets)
            return pred, targets, loss
        
        def training_step(self, batch: torch.Tensor) -> torch.Tensor:
            pred, targets, loss = self.step(batch)
            return loss
        
        def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
            pred, targets, loss = self.step(batch)
            self.log('val_loss', loss)
            return loss
        
        def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
            optimizer = self.optimizer(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            return {"optimizer": optimizer}

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int = 3, criterion: torch.nn = nn.CrossEntropyLoss(),
                 optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW, learning_rate: float = 1e-3, 
                 weight_decay: float = 1e-05, batch_size: int = 256, val_size: float = 0.2, num_workers: int = 4, 
                 max_epochs: int = 100, device_type: str = 'gpu', num_devices: int = 1, seed: int = 42):
        # used for __repr__:
        args = list(locals().items())[1:]
        defaults = {k: v for k, v in zip(list(self.__init__.__annotations__.keys())[::-1], self.__init__.__defaults__[::-1])}
        self.params = {}
        for attr, value in args:
            if attr not in defaults or defaults[attr] != value:
                self.params[attr] = value
        
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.device_type = device_type
        self.num_devices = num_devices
        self.seed = seed
        self.is_fitted = False

        torch.set_float32_matmul_precision('high')
        callbacks = [ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min', verbose=False, save_last=False), 
                     EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=False)]
        self.trainer = Trainer(strategy='auto', plugins=[LightningEnvironment()], max_epochs=self.max_epochs, 
                               accelerator=self.device_type, devices=self.num_devices, callbacks=callbacks)
        self.wrapper = self.Wrapper(criterion, optimizer, in_dim, hidden_dims, out_dim, learning_rate, weight_decay)

    def fit(self, X: np.ndarray, y):
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=self.val_size, 
                                                                          random_state=self.seed)
        train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, persistent_workers=bool(self.num_workers), 
                                  pin_memory=True)
        val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=self.batch_size, 
                                num_workers=self.num_workers, persistent_workers=bool(self.num_workers), 
                                pin_memory=True)
        self.trainer.fit(self.wrapper, train_loader, val_loader)
        self.is_fitted = True

    def predict(self, X: np.ndarray):
        dataset = SimpleDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                                persistent_workers=bool(self.num_workers), pin_memory=True)
        pred = self.trainer.predict(self.wrapper, dataloader)
        pred = torch.concat(pred, dim=0)
        pred = torch.max(pred, dim=1).indices
        pred = pred.cpu().numpy()
        return pred

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={v}" for k, v in self.params.items()])})'
