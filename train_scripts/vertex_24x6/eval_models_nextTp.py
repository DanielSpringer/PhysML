import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.analysis.vertex import PhaseClassification
from phys_ml.evaluation import vertex as verteval
from phys_ml.load_data.vertex import *

import pandas as pd

from sklearn import ensemble


if __name__ == '__main__':
    latent_dims = [8, 16, 20, 24, 32]
    ssizes = [2000, 8000, 12000, 16000]

    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)
    next_vertices = {k: v for k, v in zip(vertices.keys(), list(vertices.values())[1:])}

    # autoencoder
    seed = 42
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

    # reconstruction filepaths
    ex_sc_fps = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=dataset_kwargs['subset_shuffle'], 
                                                        subset_seed=seed, subset_type=['afm', 'fm'], file_paths=file_paths)[0]
    ex_sc_fps = list(set(file_paths) - set(ex_sc_fps))

    # test sets
    test_dataset_full = verteval.make_dataset(vertices, test_samples_per_vertex, dataset_kwargs, dataset_class=AutoEncoder24x6InfoNCEDataset)

    # run info
    base_path = '/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results/'
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/notebooks/vertex/'
    rmse_df = pd.DataFrame(columns=['run_id', 'ld', 's', 'tp', 'rmse', 'train_data'])
    classification_df = pd.DataFrame(columns=['run_id', 'ld', 's', 'f1', 'conf_mat'])
    try:
        rmse_df = pd.read_csv(save_path + 'reconstruction_results.csv')
        classification_df = pd.read_pickle(save_path + 'classification_results.pkl')
    except:
        pass

    # evaluate models
    def eval(run_id: str, run_name: str, ld: int, s: int, recon_files: list[str]):
        model_path = base_path + run_name
        if os.path.exists(model_path):
            # reconstruction
            if len(rmse_df[(rmse_df['run_id'] == run_id) 
                        & (rmse_df['ld'] == ld) 
                        & (rmse_df['s'] == s)]) < len(file_paths):
                rmses = verteval.mean_rmse(file_paths, next_vertices, model_path)
                is_train_data = [fp not in recon_files for fp in file_paths]
                for is_td, (tp, rmse) in zip(is_train_data, rmses.items()):
                    rmse_df.loc[len(rmse_df)] = [run_id, ld, s, tp, rmse, is_td]
                rmse_df.to_csv(save_path + 'reconstruction_results.csv', index=False)

            # classification
            if classification_df[(classification_df['run_id'] == run_id)
                                & (classification_df['ld'] == ld) 
                                & (classification_df['s'] == s)].empty:
                pc = PhaseClassification(model_path, pc_models, run_name)
                pc.train(nce_train_dataset)
                model_scores = pc.evaluate_classifiers(test_dataset_full, print_conf_mat=False)
                pc_results = list(model_scores.values())[0]
                classification_df.loc[len(classification_df)] = [run_id, ld, s, pc_results[0]['f1'], pc_results[1]]
                classification_df.to_pickle(save_path + 'classification_results.pkl')

    run_id = '2_1_1_nextTp'
    ld = 32
    run_name = f'{run_id}_ld{ld}'
    run_name = f'2_1_1_nextTp_ld32'
    eval(run_id, run_name, ld, 24_000, ex_sc_fps)