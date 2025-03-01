import sys, os
sys.path.append('C:/OneDrive - TU Wien/Studium/Master_5. Semester/Masters thesis/code/PhysML')

from phys_ml.trainer import TrainerModes
from phys_ml.trainer.vertex import VertexTrainer24x6


def train():
    trainer = VertexTrainer24x6('vertex_24x6', config_name='confmod_auto_encoder.json', 
                                subconfig_name='AUTO_ENCODER_VERTEX_24X6', 
                                config_kwargs={'path_train': '/gpfs/data/fs71925/shepp123/frgs_6d', 
                                               'hidden_dims': [128, 64, 32], 
                                               'sample_count_per_vertex': 2000, 'epochs': 100, 
                                               'test_ratio': 0.2, 'subset': -1, 'subset_shuffle': True, 
                                               'subset_seed': 42, 'strategy': 'auto'})
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
