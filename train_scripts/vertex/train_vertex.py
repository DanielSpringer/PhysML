import sys, os
sys.path.append(os.getcwd())

from phys_ml.trainer import TrainerModes, VertexTrainer


def train():
    trainer = VertexTrainer('vertex', 'confmod_auto_encoder.json', 'AUTO_ENCODER_VERTEX')
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
