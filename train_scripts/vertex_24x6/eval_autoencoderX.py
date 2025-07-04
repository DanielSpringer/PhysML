import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.analysis.vertex import PhaseClassification
from phys_ml.evaluation import vertex as verteval
from phys_ml.load_data.vertex import *

import pandas as pd

from sklearn import ensemble
from tqdm.notebook import tqdm


if __name__ == '__main__':
    latent_dims = [8, 16, 24, 32]
    ssizes = [2000, 8000, 16000]


    # load vertices
    path_train = '../../../frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)

    # autoencoder
    seed = 42
    hidden_dims = [128, 64, 32]
    train_samples_per_vertex = 24000
    nce_train_samples = train_samples_per_vertex // 4
    test_samples_per_vertex = 2000
    dataset_kwargs = {
        'path_train': path_train, 
        'subset_shuffle': False, 
    }

    # phase classifier
    pc_models = [
        # svm.SVC(verbose=True, random_state=seed + 1), 
        ensemble.RandomForestClassifier(n_jobs=-1, verbose=0, random_state=seed + 2),
    ]

    # general train sets
    nce_train_dataset = verteval.make_dataset(vertices, nce_train_samples, dataset_kwargs, dataset_class=AutoEncoder24x6InfoNCEDataset)
    nce_train_dataset_subset = verteval.make_dataset(vertices, nce_train_samples, dataset_kwargs, subset=0.8, 
                                                    dataset_class=AutoEncoder24x6InfoNCEDataset)

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

    # test sets
    test_dataset_full = verteval.make_dataset(vertices, test_samples_per_vertex, dataset_kwargs, dataset_class=AutoEncoder24x6InfoNCEDataset)
    test_dataset_subset = verteval.make_test_from_train_dataset(vertices, path_train, dataset_kwargs, test_samples_per_vertex, 
                                                                nce_train_dataset_subset)

    # run info
    base_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/'
    run_info = {
        # '1_1': (file_paths, nce_train_dataset, test_dataset_full), 
        # '1_2': (file_paths, nce_train_dataset, test_dataset_full), 
        '2_1_1': (ex_sc_fps, nce_train_dataset_subset, test_dataset_subset), 
        '2_1_2': (ex_sc_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '2_2_1': (ex_afm_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '2_2_2': (ex_afm_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '2_3_1': (ex_fm_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '2_3_2': (ex_fm_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '3_1': (sc_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '3_2': (afm_fps, nce_train_dataset_subset, test_dataset_subset), 
        # '3_3': (fm_fps, nce_train_dataset_subset, test_dataset_subset),
    }
    reconstruction_results = []
    classification_results = []


    # evaluate models
    def eval(prog: tqdm, run_name: str, ld: int, s: int):
        prog.set_description(f'Evaluating {run_name}')
        save_path = base_path + run_name

        # reconstruction
        rmses = verteval.mean_rmse(file_paths, vertices, save_path)
        is_train_data = [fp not in recon_files for fp in file_paths]
        reconstruction_results.extend([
            {'run_id': run_id, 'ld': ld, 's': s, 'tp': tp, 'rmse': rmse, 'train_data': is_td} 
            for is_td, (tp, rmse) in zip(is_train_data, rmses.items())
        ])

        # classification
        pc = PhaseClassification(save_path, pc_models, run_name)
        pc.train(nce_train_dataset)
        model_scores = pc.evaluate_classifiers(test_dataset_full, print_conf_mat=False)
        pc_results = list(model_scores.values())[0]
        classification_results.append({'run_id': run_id, 'ld': ld, 's': s, 'f1': pc_results[0]['f1'], 'conf_mat': pc_results[1]})
        prog.update()


    total = len(run_info) * len(latent_dims)
    with tqdm(total=total, desc='Evaluation') as prog:
        for run_id, (recon_files, train_data, test_data) in run_info.items():
            for ld in latent_dims:
                run_name = f'{run_id}_ld{ld}'
                eval(prog, run_name, ld, 24_000)
            for s in ssizes:
                run_name = f'{run_id}_s{s}'
                eval(prog, run_name, 32, s)

    rmse_df = pd.DataFrame(reconstruction_results)
    rmse_df.to_csv(base_path + 'reconstruction_results.csv', index=False)
    classification_df = pd.DataFrame(classification_results)
    classification_df.to_csv(base_path + 'classification_results.csv', index=False)
