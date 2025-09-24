import sys, os
sys.path.append('C:/OneDrive - AIT/Studium/Master_5. Semester/Masters thesis/code/PhysML')

from phys_ml.trainer import TrainerModes
from phys_ml.trainer.vertex import VertexTrainer24x6


def train():
    trainer = VertexTrainer24x6('vertex_24x6', config_name='confmod_auto_encoder.json', subconfig_name='AUTO_ENCODER_VERTEX_24X6', config_kwargs={'path_train': '/gpfs/data/fs71925/shepp123/frgs_6d', 'resume': 'run_results_2/1_2', 'model_wrapper': 'VertexWrapper24x6InfoNCE', 'model_name': 'ContrastiveAutoEncoder', 'dataset': 'AutoEncoder24x6InfoNCEDataset', 'hidden_dims': [128, 64, 32], 'epochs': 50000, 'sample_count_per_vertex': 6000, 'test_ratio': 0.2, 'subset': None, 'subset_type': None, 'subset_shuffle': True, 'sample_seed': 123, 'subset_seed': 123, 'devices': 'auto', 'device_type': 'gpu', 'num_dataloader_workers': 2, 'batch_size': 2048})
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
