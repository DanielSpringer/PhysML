import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    verteval.evaluate_all('3_2', 32, 24000, 'run_results_2', 123, models=['rf'])
