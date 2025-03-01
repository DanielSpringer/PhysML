import sys, os
sys.path.append(os.getcwd())

from phys_ml.trainer import TrainerModes
from phys_ml.trainer.vertex import VertexTrainer


def train():
    trainer = VertexTrainer('vertex', project_name='vertex', config_name='confmod_auto_encoder.json', subconfig_name='AUTO_ENCODER_VERTEX', test=1)
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
