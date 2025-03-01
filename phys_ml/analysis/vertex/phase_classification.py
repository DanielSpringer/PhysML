import glob
import os
import pickle
import random

from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn import ensemble, metrics, svm, model_selection as ms
from sklearn.utils.validation import check_is_fitted
from tqdm.notebook import tqdm

from phys_ml.trainer.vertex import VertexTrainer24x6


class PhaseClassification:
    def __init__(self, data_dir: str, model_path: str, samples_per_vertex: int, test_size: int|float = 0.1, 
                 latent_space_paths: list[str]|None = None, shuffle: bool = True, seed: int = 12):
        self.samples_per_vertex = samples_per_vertex
        self.test_size = test_size
        self.file_type: Literal['vertex', 'latent_space'] = None
        if not latent_space_paths:
            file_paths = glob.glob(os.path.join(data_dir, '*.h5'))
            self.file_type = 'vertex'
        else:
            file_paths = latent_space_paths
            self.file_type = 'latent_space'
        self.file_paths, self.predict_paths = ms.train_test_split(file_paths, test_size=test_size, shuffle=shuffle, 
                                                                  random_state=seed)
        self.predict_samples_per_vertex = (test_size if isinstance(test_size, int) else 
                                           int(samples_per_vertex / (1 - test_size) * test_size))
        self.vertex_trainer = VertexTrainer24x6(project_name='vertex_24x6', load_from=model_path)
        _ = self.vertex_trainer.load_model(load_from=model_path, predict=True, encode_only=True)
        self.device = self.vertex_trainer.get_device_from_accelerator(self.vertex_trainer.config.device_type)
        self.random_idx_generator = random.Random(seed)
        self.models = [
            svm.SVC(verbose=True, random_state=seed + 1), 
            ensemble.RandomForestClassifier(n_jobs=-1, verbose=1, random_state=seed + 2),
        ]
        self.load_models()
        self.ls_length = self.vertex_trainer.config.hidden_dims[-1]
        self.predict_samples: tuple[np.ndarray, list[int]] = None
    
    def get_phase_from_filepath(self, file_path: str) -> int:
        fname = Path(file_path).stem
        tp, mu = (float(s[2:]) for s in fname.split('_'))
        return 2 if tp > 0.35 else 1 if tp > 0.15 else 0
    
    def predict_latent_space_vectors(self, vertex: np.ndarray, samples_per_vertex: int) -> np.ndarray:
        # NOTE: also possible with dataloader and trainer.predict
        input_vectors, input_idcs = self.vertex_trainer.dataset.sample(vertex, samples_per_vertex)
        input_vectors = torch.tensor(input_vectors, dtype=torch.float32).to(self.device)
        ls_vectors = self.vertex_trainer.wrapper.predict_step((input_vectors, input_idcs)).detach().cpu().numpy()
        return ls_vectors

    def load_data(self, predict_only: bool = False) -> tuple[np.ndarray, list[int], tuple[np.ndarray, list[int]]]:
        inputs = np.empty((0, self.ls_length))
        test_inputs = np.empty((0, self.ls_length))
        targets = []
        test_targets = []
        n_train_samples = 0 if predict_only else self.samples_per_vertex
        n_test_samples = self.predict_samples_per_vertex
        for fp in tqdm(self.file_paths, desc='Load data'):
            phase = self.get_phase_from_filepath(fp)
            if self.file_type == 'vertex':
                vertex = self.vertex_trainer.dataset.load_from_file(fp)
                ls_vectors = self.predict_latent_space_vectors(vertex, n_train_samples + n_test_samples)
            else:
                ls_vectors = np.load(fp)
            test_inputs = np.concatenate((test_inputs, ls_vectors[n_train_samples:]), axis=0)
            test_targets.extend([phase] * n_test_samples)
            if not predict_only:
                inputs = np.concatenate((inputs, ls_vectors[:n_train_samples]), axis=0)
                targets.extend([phase] * n_train_samples)
        return inputs, targets, (test_inputs, test_targets)

    def train(self) -> dict[str, np.ndarray]:
        inputs, targets, self.predict_samples = self.load_data()
        for model in tqdm(self.models, desc='Fit models'):
            model.fit(inputs, targets)
            with open(f'models/{model.__class__.__qualname__}.pkl','wb') as f:
                pickle.dump(model,f)

    def load_models(self):
        for i, model in enumerate(self.models):
            try:
                check_is_fitted(model)
            except:
                try:
                    with open(f'models/{model.__class__.__qualname__}.pkl', 'rb') as f:
                        self.models[i] = pickle.load(f)
                except:
                    raise ValueError(f'Model {model.__class__.__qualname__} is not fitted and could not be loaded.')

    def evaluate_model(self, model, test_ls_vectors: np.ndarray, test_targets: np.ndarray,
                    print_conf_mat: bool = True) -> tuple[dict[str, float], np.ndarray]:
        pred = model.predict(test_ls_vectors)
        scores = {
            'acc': metrics.balanced_accuracy_score(test_targets, pred),
            'prec': metrics.precision_score(test_targets, pred, average='micro'),
            'rec': metrics.recall_score(test_targets, pred, average='micro'),
            'f1': metrics.f1_score(test_targets, pred, average='micro'),
        }
        conf_mat = metrics.confusion_matrix(test_targets, pred, normalize='all')
        if print_conf_mat:
            #conf_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
            #conf_disp.plot(cmap=plt.cm.PiYG, values_format=".2f")  # vanimo, coolwarm
            sns.heatmap(conf_mat, annot=True, fmt=".2f")
            plt.show()
        return scores, conf_mat

    def evaluate_classifiers(self, print_conf_mat: bool = True) -> dict[str, tuple[dict[str, float], np.ndarray]]:
        self.load_models()
        if not self.predict_samples:
            _, _, self.predict_samples = self.load_data(predict_only=True)
        
        # using new samples from known vertices:
        known_vertices = {}
        for model in tqdm(self.models, desc='Predict with sampling from known vertices'):
            scores, conf_mat = self.evaluate_model(model, *self.predict_samples, print_conf_mat)
            known_vertices[model.__class__.__qualname__] = (scores, conf_mat)
        
        # using samples from new vertex:
        new_vertices = {}
        for model in tqdm(self.models, desc='Predict with sampling from new vertices'):
            inputs = np.empty((0, self.ls_length))
            targets = []
            for fp in tqdm(self.predict_paths, desc='Load data', leave=False):
                phase = self.get_phase_from_filepath(fp)
                if self.file_type == 'vertex':
                    vertex = self.vertex_trainer.dataset.load_from_file(fp)
                    ls_vectors = self.predict_latent_space_vectors(vertex, self.samples_per_vertex)
                else:
                    ls_vectors = np.load(fp)
                inputs = np.concatenate((inputs, ls_vectors), axis=0)
                targets.extend([phase] * self.samples_per_vertex)
            scores, conf_mat = self.evaluate_model(model, inputs, targets, print_conf_mat)
            new_vertices[model.__class__.__qualname__] = (scores, conf_mat)
        return {'sampling from new vertices': new_vertices, 'sampling from known vertices': known_vertices}
