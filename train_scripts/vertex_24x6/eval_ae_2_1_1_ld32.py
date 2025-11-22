import sys
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.evaluation import vertex as verteval


if __name__ == '__main__':
    print(__file__)
    verteval.evaluate_all('2_1_1', 32, 24000, None, 123, models=[None, None, None, None, None, None, None, None, 'spline_k40', 'spline_k100', 'spline_k250'], device='cpu')
    # verteval.evaluate_all('2_1_1', 32, 24000, 'run_results_2', 123, models=['rf', 'gp', 'nn', 'rfr', 'poly', 'svc', 'spline', 'spline_k10', 'spline_k40', 'spline_k100', 'spline_k250'])
