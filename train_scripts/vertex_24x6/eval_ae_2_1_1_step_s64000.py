import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    print(__file__)
    verteval.evaluate_all('2_1_1_step', 32, 64000, 'run_results_2', 123, models=[])
