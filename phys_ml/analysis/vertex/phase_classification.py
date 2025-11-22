import logging
import pickle
import re

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from tqdm.notebook import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader

from phys_ml.load_data.base import SimpleDataset
from phys_ml.load_data.vertex import AutoEncoderVertex24x6Dataset, AutoEncoder24x6InfoNCEDataset
from phys_ml.trainer.vertex import VertexTrainer24x6
from phys_ml.visualization import vertex_visualization as vertvis
from phys_ml.util import is_notebook


class PhaseClassification:
    MODEL_SAVE_PATTERN = '%s_%s_%s.pkl'

    def __init__(self, model_path: str, models: list[BaseEstimator], version: str, batch_size: int = 2048, 
                 device: Literal['cpu', 'cuda']|None = None):
        self.models = models
        self.version = version
        self.encode = model_path is not None
        self.logger = logging.getLogger('lightning.pytorch')
        if self.encode:
            self.vertex_trainer = VertexTrainer24x6(project_name='vertex_24x6', load_from=model_path)
            _ = self.vertex_trainer.load_model(load_from=model_path, load_ckpt=None, predict=True, encode_only=True)
            self.ls_length = self.vertex_trainer.config.hidden_dims[-1]
            self.device = device or self.vertex_trainer.get_device_from_accelerator(self.vertex_trainer.config.device_type)
            self.batch_size = self.vertex_trainer.config.batch_size
        else:
            self.ls_length = AutoEncoderVertex24x6Dataset.length * AutoEncoderVertex24x6Dataset.dim
            self.batch_size = batch_size
        self.predict_samples: tuple[np.ndarray, list[int]] = None
    
    def get_phase_from_filepath(self, file_path: str) -> tuple[float, int]:
        fname = Path(file_path).stem
        tp, mu = (float(s[2:]) for s in fname.split('_'))
        for i, (phase, borders) in enumerate(AutoEncoderVertex24x6Dataset.phase_borders.items()):
            if tp >= borders[0] and tp < borders[1]:
                return i, tp

    def load_data(self, dataset: AutoEncoderVertex24x6Dataset) -> tuple[np.ndarray, list[int], list[float]]:
        inputs = np.empty((0, self.ls_length))
        phase_labels = []
        tp_labels = []
        desc = 'Encode vertex samples' if self.encode else 'Load vertex samples'
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        iterator = dataloader
        if is_notebook():
            iterator = tqdm(iterator, desc=desc, leave=False)
        for input_vectors, idcs, _, fps in iterator:
            if isinstance(dataset, AutoEncoder24x6InfoNCEDataset):
                input_vectors = input_vectors.reshape((-1, input_vectors.shape[-1]))
                idcs = idcs.reshape((-1, idcs.shape[-1]))
                fps = [fp for sublist in zip(*fps) for fp in sublist]
            phases, tps = zip(*[self.get_phase_from_filepath(fp) for fp in fps])
            phase_labels.extend(phases)
            tp_labels.extend(tps)
            if self.encode:
                input_vectors = input_vectors.to(self.device)
                ls_vectors = self.vertex_trainer.wrapper.predict_step((input_vectors, idcs))
                ls_vectors = ls_vectors.detach()
                ls_vectors = ls_vectors.cpu()
                ls_vectors = ls_vectors.numpy()
            else:
                ls_vectors = input_vectors
            inputs = np.concatenate((inputs, ls_vectors), axis=0)
        return inputs, phase_labels, tp_labels
    
    def is_regression_model(self, model: BaseEstimator) -> bool:
        return 'regress' in model.__class__.__name__.lower()

    def train(self, dataset: AutoEncoderVertex24x6Dataset):
        inputs, phase_labels, tp_labels = self.load_data(dataset)
        iterator = self.models
        if is_notebook():
            iterator = tqdm(iterator, desc='Fit models', leave=False)
        for i, model in enumerate(iterator):
            try:
                _ = self.train_model(i, model, inputs, phase_labels, tp_labels)
            except Exception as e:
                self.logger.info(f'Could not fit model {repr(model)}:\n\t{e}')

    def train_model(self, i: int, model: BaseEstimator, inputs:np.ndarray, phase_labels: list[int], 
                    tp_labels: list[float]) -> BaseEstimator:
        self.logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Fit model {repr(model)}')
        targets = tp_labels if self.is_regression_model(model) else phase_labels
        if type(model) == NeuralNetClassifier:
            model.init_model(self.ls_length)
        model.fit(inputs, targets)
        with open(self.MODEL_SAVE_PATTERN % (self.version, f'{i:02}', model.__class__.__name__), 'wb') as f:
            pickle.dump(model,f)
        self.logger.info(
            f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Finished training model {repr(model)} ({i + 1} of {len(self.models)})')
        self.models[i] = model
        return model

    def load_models(self, verbose: bool = False) -> list[bool]:
        is_loaded = []
        for i, model in enumerate(self.models):
            try:
                check_is_fitted(model)
            except:
                try:
                    with open(self.MODEL_SAVE_PATTERN % (self.version, f'{i:02}', model.__class__.__name__), 'rb') as f:
                        self.models[i] = pickle.load(f)
                except:
                    if verbose:
                        self.logger.info(f'Model {repr(model)} is not fitted and could not be loaded. '
                                         'Call train() before using the PhaseClassifier.')
                    is_loaded.append(False)
                    continue
            is_loaded.append(True)
        return is_loaded

    def evaluate_model(self, i: int, model: BaseEstimator, test_ls_vectors: np.ndarray, phase_targets: list[int], 
                       tp_targets: list[float], labels: list[str], print_conf_mat: bool = True, 
                       normalize: Literal['true', 'pred', 'all'] = 'true') -> tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Evaluate model {repr(model)}')
        pred: np.ndarray = model.predict(test_ls_vectors)
        model_name = repr(model)
        class_data = {'model': model_name}
        reg_data = []
        is_regression_model = self.is_regression_model(model)
        if is_regression_model:
            class_data['rmse'] = metrics.root_mean_squared_error(tp_targets, pred)
            pmin, pmax = 0.0, 0.5
            for p, tp in zip(pred, tp_targets):
                reg_data.append({'model': model_name, 'pred': p, 'tp': tp})
            pred = np.clip(pred, pmin, pmax)
            pred = [i for p in pred for i, v in enumerate(AutoEncoderVertex24x6Dataset.phase_borders.values()) 
                    if p >= v[0] and p < v[1]]
        class_data.update({
            'acc': metrics.balanced_accuracy_score(phase_targets, pred),
            # 'prec': metrics.precision_score(test_targets, pred, average='micro'),
            # 'rec': metrics.recall_score(test_targets, pred, average='micro'),
            'f1': metrics.f1_score(phase_targets, pred, average='micro'),
        })
        if not is_regression_model:
            class_data['rmse'] = np.nan
        labels = np.array(labels)
        pred = labels[pred]
        phase_targets = labels[phase_targets]
        conf_mat = metrics.confusion_matrix(phase_targets, pred, labels=labels, normalize=normalize)
        class_data['conf_mat'] = conf_mat
        if print_conf_mat:
            vertvis.print_conf_mat(conf_mat, model.__class__.__name__, labels)
        
        # save results
        classification_df = pd.DataFrame([class_data])
        regression_df = pd.DataFrame(reg_data)
        run_parameters = self.version.split('_')
        run_id = '_'.join(run_parameters[:-1])
        param, pvalue = re.split(r'([a-zA-Z]+)(\d+)', run_parameters[-1])[1:3]
        ld, s, seed = 32, 24000, 123
        match param:
            case 'ld':
                ld = int(pvalue)
            case 's':
                s = int(pvalue)
            case 'seed':
                seed = int(pvalue)
            case _:
                pass
        classification_df[['run_id', 'ld', 's', 'seed']] = [run_id, ld, s, seed]
        regression_df[['run_id', 'ld', 's', 'seed']] = [run_id, ld, s, seed]
        classification_df.to_pickle(f'classification_results_{self.version}_{i:02}_{model.__class__.__name__}.pkl')
        regression_df.to_csv(f'regression_results_{self.version}_{i:02}_{model.__class__.__name__}.csv', index=False)
        
        self.logger.info(
            f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Finished evaluating model {repr(model)} ({i + 1} of {len(self.models)})')
        return classification_df, regression_df

    def evaluate_classifiers(self, dataset: AutoEncoderVertex24x6Dataset, train_dataset: AutoEncoderVertex24x6Dataset|None, 
                             print_conf_mat: bool = True, 
                             normalize: Literal['true', 'pred', 'all'] = 'true') -> tuple[pd.DataFrame, pd.DataFrame]:
        classification_dfs = []
        regression_dfs = []
        models_loaded = self.load_models(verbose=True)
        if not all(models_loaded):
            train_inputs, train_phase_labels, train_tp_labels = self.load_data(train_dataset)
        test_inputs, test_phase_labels, test_tp_labels = self.load_data(dataset)
        labels = list(dataset.phase_borders.keys())
        iterator = models_loaded
        if is_notebook():
            iterator = tqdm(iterator, desc='Predict', leave=False)
        for i, is_loaded in enumerate(iterator):
            try:
                if is_loaded:
                    model = self.models[i]
                else:
                    model = self.train_model(i, self.models[i], train_inputs, train_phase_labels, train_tp_labels)
                c_df, r_df = self.evaluate_model(i, model, test_inputs, test_phase_labels, test_tp_labels, labels, 
                                                 print_conf_mat, normalize)
                classification_dfs.append(c_df)
                regression_dfs.append(r_df)
            except Exception as e:
                self.logger.info(e)
                self.logger.info(f'Skipping model {repr(self.models[i])}, which could not be fitted.')
        classification_df = pd.concat(classification_dfs)
        regression_df = pd.concat(regression_dfs)
        return classification_df, regression_df


class StandardSVC(SVC):
    def fit(self, X, y, sample_weight = None):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        return super().fit(X, y, sample_weight)
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return super().predict(X)


class PolynomialRegression(LinearRegression):
    def __init__(self, fit_intercept: bool = False, copy_X: bool = True, n_jobs: int|None = -1, positive: bool = True, 
                 dtype = np.float32):
        self.polynomial: PolynomialFeatures|SplineTransformer = None
        self.params: dict[str, Any] = None
        self.scaler = StandardScaler()
        self.dtype = dtype
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
    
    @classmethod
    def new(cls, method: Literal['poly', 'spline'] = 'poly', degree: int|tuple[int, int] = 2, 
            n_knots: int = 10, interaction_only: bool = False, include_bias: bool = False, order: Literal['C', 'F'] = 'C', 
            fit_intercept: bool = False, copy_X: bool = True, n_jobs: int|None = -1, positive: bool = True) -> 'PolynomialRegression':
        # used for __repr__:
        defaults = {k: v for k, v in zip(list(cls.new.__annotations__.keys())[:-1], cls.new.__defaults__)}
        params = {}
        for attr, value in defaults.items():
            if (param := locals()[attr]) != value:
                params[attr] = param
        if method == 'poly':
            polynomial = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
        elif method == 'spline':
            polynomial = SplineTransformer(degree=degree, n_knots=n_knots)
        model = cls(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
        model.polynomial = polynomial
        model.params = params
        return model
    
    def fit(self, X: np.ndarray, y, sample_weight = None):
        X = X.astype(self.dtype)
        self.scaler = self.scaler.fit(X)
        X = self.scaler.transform(X)
        polynomials = self.polynomial.fit_transform(X)
        return super().fit(polynomials, y, sample_weight)
    
    def predict(self, X: np.ndarray):
        X = X.astype(self.dtype)
        X = self.scaler.transform(X)
        polynomials = self.polynomial.transform(X)
        return super().predict(polynomials)
    
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

    def __init__(self, in_dim: int|None = None, hidden_dims: list[int]|None = None, n_layers: int = 3, out_dim: int = 3, 
                 criterion: torch.nn = nn.CrossEntropyLoss(), optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-05, batch_size: int = 256, val_size: float = 0.2, 
                 num_workers: int = 4, max_epochs: int = 100, device_type: str = 'gpu', num_devices: int = 1, seed: int = 42):
        # used for __repr__:
        args = list(locals().items())[1:]
        defaults = {k: v for k, v in zip(list(self.__init__.__annotations__.keys())[::-1], self.__init__.__defaults__[::-1])}
        self.params = {}
        for attr, value in args:
            if attr not in defaults or defaults[attr] != value:
                self.params[attr] = value
        
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.device_type = device_type
        self.num_devices = num_devices
        self.seed = seed
        self.is_fitted = False

        torch.set_float32_matmul_precision('high')
        callbacks = [ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min', verbose=True, save_last=False), 
                     EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True)]
        self.trainer = Trainer(strategy='ddp', max_epochs=self.max_epochs, accelerator=self.device_type, 
                               devices=self.num_devices, callbacks=callbacks, enable_progress_bar=False, enable_model_summary=False)
        if in_dim is not None and hidden_dims is not None:
            self.init_model(in_dim, hidden_dims)
        else:
            self.wrapper = None
    
    def init_model(self, in_dim: int, hidden_dims: list[int]|None = None):
        if hidden_dims is None:
            hidden_dims = (np.linspace(in_dim, self.out_dim, self.n_layers + 1, dtype=int)[1:-1] // 2 * 2).tolist()
        self.wrapper = self.Wrapper(self.criterion, self.optimizer, in_dim, hidden_dims, self.out_dim, self.learning_rate, 
                                    self.weight_decay)

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
        pred = torch.cat(pred, dim=0)
        pred = torch.max(pred, dim=1).indices
        pred = pred.cpu().numpy()
        return pred

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={v}" for k, v in self.params.items()])})'
