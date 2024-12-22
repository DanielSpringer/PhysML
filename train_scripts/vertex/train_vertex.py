import sys, os
sys.path.append(os.getcwd())

from phys_ml.trainer import TrainerModes, VertexTrainer24x6


def train():
    trainer = VertexTrainer24x6('vertex', project_name='vertex', config_name='confmod_auto_encoder.json', subconfig_name='AUTO_ENCODER_VERTEX_24X6', test=1)
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
