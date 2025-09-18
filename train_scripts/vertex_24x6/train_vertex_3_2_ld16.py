import sys, os
sys.path.append('/gpfs/data/fs71925/shepp123/PhysML')

from phys_ml.trainer import TrainerModes
from phys_ml.trainer.vertex import VertexTrainer24x6


def train():
    trainer = VertexTrainer24x6('vertex_24x6', config_name='confmod_auto_encoder.json', subconfig_name='AUTO_ENCODER_VERTEX_24X6', config_kwargs={'path_train': '/gpfs/data/fs71925/shepp123/frgs_6d', 'resume': 'run_results_2/3_2_ld16', 'model_wrapper': 'VertexWrapper24x6', 'model_name': 'AutoEncoderVertex', 'dataset': 'AutoEncoderVertex24x6Dataset', 'hidden_dims': [128, 64, 32, 16], 'epochs': 25000, 'sample_count_per_vertex': 24000, 'test_ratio': 0.2, 'subset': 0.8, 'subset_type': 'afm', 'subset_shuffle': True, 'subset_seed': 123, 'devices': 'auto', 'device_type': 'gpu', 'num_dataloader_workers': 2, 'batch_size': 8192})
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
