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
    seed = 12
    hidden_dims = [128, 64, 32, 8]
    config_kwargs = {
        'hidden_dims': hidden_dims,
        'epochs': 1000,
        'test_ratio': 0.2, 
        'devices': 'auto', 
        'num_dataloader_workers': 16, 
        'strategy': 'auto', 
        'batch_size': 8192 * 2,
    }
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': False, 
    }

    # reconstruction filepaths
    ex_sc_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                       subset_seed=seed, subset_type=['afm', 'fm'], file_paths=file_paths)[0]
    ex_sc_fps = list(set(file_paths) - set(ex_sc_fps))
    ex_afm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                            subset_seed=seed, subset_type=['sc', 'fm'], file_paths=file_paths)[0]
    ex_afm_fps = list(set(file_paths) - set(ex_afm_fps))
    ex_fm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type=['afm', 'sc'], file_paths=file_paths)[0]
    ex_fm_fps = list(set(file_paths) - set(ex_fm_fps))
    sc_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='sc', file_paths=file_paths)[0]
    sc_fps = list(set(file_paths) - set(sc_fps))
    afm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='afm', file_paths=file_paths)[0]
    afm_fps = list(set(file_paths) - set(afm_fps))
    fm_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type='fm', file_paths=file_paths)[0]
    fm_fps = list(set(file_paths) - set(fm_fps))
    
    # # 1-1
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/1_1_8'
    # preds = verteval.predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, encode_only=False,
    #                              train_mode=TrainerModes.SLURM)

    # # 1-2
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/1_2_8'
    # preds = verteval.predict_all(file_paths, vertices, save_path, config_kwargs, dataset_kwargs, encode_only=False,
    #                              train_mode=TrainerModes.SLURM)
    
    # 2-1-1
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_1_1_8'
    preds = verteval.predict_all(ex_sc_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
                                encode_only=False, subset_type=['afm', 'fm'], train_mode=TrainerModes.SLURM)

    # 2-1-2
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_1_2_8'
    preds = verteval.predict_all(ex_sc_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
                                encode_only=False, subset_type=['afm', 'fm'], train_mode=TrainerModes.SLURM)

    # # 2-2-1
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_2_1_8'
    # preds = verteval.predict_all(ex_afm_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
    #                             encode_only=False, subset_type=['sc', 'fm'], train_mode=TrainerModes.SLURM)

    # # 2-2-2
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_2_2_8'
    # preds = verteval.predict_all(ex_afm_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
    #                             encode_only=False, subset_type=['sc', 'fm'], train_mode=TrainerModes.SLURM)

    # # 2-3-1
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_3_1_8'
    # preds = verteval.predict_all(ex_fm_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
    #                             encode_only=False, subset_type=['afm', 'sc'], train_mode=TrainerModes.SLURM)

    # # 2-3-2
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/2_3_2_8'
    # preds = verteval.predict_all(ex_fm_fps, vertices, save_path, config_kwargs, dataset_kwargs, 
    #                             encode_only=False, subset_type=['afm', 'sc'], train_mode=TrainerModes.SLURM)

    # # 3-1
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/3_1_8'
    # preds = verteval.predict_all(sc_fps, vertices, save_path, config_kwargs, dataset_kwargs, encode_only=False, subset_type='sc',
    #                              train_mode=TrainerModes.SLURM)

    # # 3-2
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/3_2_8'
    # preds = verteval.predict_all(afm_fps, vertices, save_path, config_kwargs, dataset_kwargs, encode_only=False, subset_type='afm',
    #                              train_mode=TrainerModes.SLURM)

    # # 3-3
    # save_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/3_3_8'
    # preds = verteval.predict_all(fm_fps, vertices, save_path, config_kwargs, dataset_kwargs, encode_only=False, subset_type='fm',
    #                              train_mode=TrainerModes.SLURM)
