import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    verteval.predict_for_all_models('2_1_1', 32, 24000, 'run_results_3', 89)
