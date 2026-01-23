import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from pathlib import Path

from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    print(__file__)
    verteval.predict_for_all_models('2_1_1_step', 32, 64000, 'run_results_2', 123)
    pred_path = Path('/gpfs/data/fs71925/shepp123/PhysML/saves/vertex_24x6/run_results_2/2_1_1_step_s64000/predictions')
    filepaths = reversed(sorted([p for p in pred_path.glob('tp*')]))
    filenames = [p.name for p in filepaths]
    for i, fp in enumerate(filepaths):
        if i == 0:
            fp.rename(pred_path / ('_' + filenames[i]))
        else:
            fp.rename(pred_path / filenames[i - 1])
