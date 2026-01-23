import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from pathlib import Path

from phys_ml.load_data.vertex import AutoEncoderVertex24x6Dataset
from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    print(__file__)
    run_dir_name = 'run_results_2'
    run_id = '2_1_1_step'
    ld = 32
    s = 64000
    seed = 123

    # run info
    base_path = Path('/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6')
    save_path = '/gpfs/data/fs71925/shepp123/PhysML/notebooks/vertex/eval_results/'

    # load vertices
    path_train = '/gpfs/data/fs71925/shepp123/frgs_6d'
    file_paths = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=None, subset_shuffle=False)[0]
    vertices = AutoEncoderVertex24x6Dataset.load_vertex_files(file_paths)
    train_files = AutoEncoderVertex24x6Dataset.get_filepaths(path_train, subset=0.8, subset_shuffle=True, 
                                                            subset_seed=seed, subset_type=['afm', 'fm'], file_paths=file_paths)[0]

    # evaluate models
    if s != 24000:
        pref = f's{s}'
    elif seed != 123:
        pref = f'r{seed}'
    else:
        pref = f'ld{ld}'
    run_name = f'{run_id}_{pref}'

    # reconstruction
    model_path = base_path / run_dir_name / run_name
    rmse_df = verteval.mean_rmse(vertices, model_path, train_files)
    rmse_df[['run_id', 'ld', 's', 'seed']] = [run_id, ld, s, seed]
    rmse_df.to_csv(save_path + f'reconstruction_results_{run_name}.csv', index=False)
