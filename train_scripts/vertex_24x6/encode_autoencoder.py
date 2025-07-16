import sys, os
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.evaluation import vertex as verteval
from phys_ml.load_data.vertex import *
from phys_ml.trainer import TrainerModes


if __name__ == '__main__':
    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)

    # autoencoder
    hidden_dims = [
        # [128, 64, 32, 8],
        # [128, 64, 32, 16],
        # [128, 64, 32, 20],
        # [128, 64, 24],
        [128, 64, 32],
    ]
    ssizes = [2000, 8000, 12000, 16000]
    seed = 12
    config_kwargs = {
        'hidden_dims': None,
        'epochs': 1000,
        'test_ratio': 0.2, 
        'devices': 1, 
        'num_dataloader_workers': 2, 
        'strategy': 'auto', 
        'batch_size': 8192 * 2,
    }
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': False, 
    }
    pred_configs = {
        # '1_1': {},
        # '1_2': {},
        '2_1_1': {'subset_type': ['afm', 'fm']},
        # '2_1_2': {'subset_type': ['afm', 'fm']},
        # '2_2_1': {'subset_type': ['sc', 'fm']},
        # '2_2_2': {'subset_type': ['sc', 'fm']},
        # '2_3_1': {'subset_type': ['afm', 'sc']},
        # '2_3_2': {'subset_type': ['afm', 'sc']},
        # '3_1': {'subset_type': 'sc'},
        # '3_2': {'subset_type': 'afm'},
        # '3_3': {'subset_type': 'fm'},
    }

    for run_id, pred_config in pred_configs.items():
        for hidden_dim in hidden_dims:
            ld = hidden_dim[-1]
            save_path = f'/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/{run_id}_ld{ld}'
            if os.path.exists(save_path):
                config_kwargs['hidden_dims'] = hidden_dim
                preds = verteval.predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, 
                                             encode_only=True, train_mode=TrainerModes.SLURM, **pred_config)
        # for s in ssizes:
        #     save_path = f'/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/{run_id}_s{s}'
        #     if os.path.exists(save_path):
        #         config_kwargs['hidden_dims'] = [128, 64, 32]
        #         preds = verteval.predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, 
        #                                     encode_only=False, train_mode=TrainerModes.SLURM, **pred_config)
