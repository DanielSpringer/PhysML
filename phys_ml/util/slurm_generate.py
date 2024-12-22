import importlib
import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config


@dataclass
class SlurmOptions:
    mail_type: str = 'BEGIN'
    mail_user: str = '<email@address.at>'
    qos: str = 'zen3_0512_a100x2_devel'                               # see available resources on VSC via `sqos`-command
    time: str = '00:10:00'                                            # must be <= '00:10:00' for '_devel' nodes
    
    @property
    def partition(self) -> str:
        return self.qos.removesuffix('_devel')


def create_train_script(project_name: str, script_name: str, base_dir: str|Path, trainer: str|None = None, 
                        trainer_kwargs: dict[str, Any]|None = None) -> None:
    extra_kwargs = trainer_kwargs.pop('config_kwargs', {})
    trainer_kwargs.update(extra_kwargs)
    args_string = ', '.join([f'{k}={repr(v)}' for k, v in trainer_kwargs.items()])

    s = f"""import sys, os
sys.path.append(os.getcwd())

from phys_ml.trainer import TrainerModes, {trainer}


def train():
    trainer = {trainer}('{project_name}', {args_string})
    trainer.train(train_mode=TrainerModes.SLURM)


if __name__ == '__main__':
    train()
"""
    fdir = Path(base_dir, 'train_scripts', project_name)
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f'train_{script_name}.py').write_text(s)


def create(project_name: str, script_name: str, pyenv_dir: str,
           slurm_options: SlurmOptions = SlurmOptions(), train_script_name: str|None = None, 
           trainer: str|None = None, trainer_kwargs: dict[str, Any]|None = None) -> None:
    """
    Create a slurm script and optionally create a python train-script.

    Parameters
    ----------
    project_name : str
        A name for the project.
    script_name : str
        A name for the generated slurm-file.
    pyenv_dir : str
        Path to the directory of the python environment.\n
        Given directory should contain the `/bin/` folder.
    slurm_options : SlurmOptions, optional
        `SlurmOptions`-instance containing the SBATCH options.
    train_script_name : str | None, optional
        Name of the train-script in the `train_scripts` folder to use.\n
        If None, generate a train-script. (defaults to None)
    trainer : str | None, optional
        Name of trainer-class to use in the train-script.\n
        Only required if `train_script_name` is not given. (defaults to None)
    trainer_kwargs : dict[str, Any] | None, optional
        Kwargs for instantiating the trainer-class in the train-script.\n
        Only required if `train_script_name` is not given. (defaults to None)
    """
    config_cls: type[Config] = getattr(importlib.import_module('phys_ml.trainer'), trainer).config_cls
    config_kwargs = {k: trainer_kwargs[k] 
                     for k in ['config_name', 'subconfig_name', 'config_dir', 'config_kwargs'] 
                     if k in trainer_kwargs}
    config = config_cls.from_json(**config_kwargs)
    venv_files = Path(pyenv_dir, '*').as_posix()
    source_path = Path(pyenv_dir, 'bin/activate').as_posix()
    current_base_dir = Path(__file__).parent.parent.parent
    
    if not train_script_name:
        create_train_script(project_name, script_name, current_base_dir, trainer, trainer_kwargs)
        train_script_name = f'train_{script_name}.py'
    train_script_path = (current_base_dir / 'train_scripts' / project_name / train_script_name).as_posix()
    
    s = f"""#!/bin/bash
#
#SBATCH -J {project_name}
#SBATCH --mail-type={slurm_options.mail_type}    # first have to state the type of event to occur 
#SBATCH --mail-user={slurm_options.mail_user}    # and then your email address

#SBATCH --partition={slurm_options.partition}
#SBATCH --qos={slurm_options.qos}
#SBATCH --ntasks-per-node={config.devices // config.num_nodes}
#SBATCH --nodes={config.num_nodes}
#SBATCH --time={slurm_options.time}

FILES=({venv_files})
source {source_path}

srun uv run {train_script_path}
"""
    fdir = current_base_dir / 'slurm' / project_name
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f'vsc_{script_name}.slrm').write_text(s)


def create_from_config(config_file: str|Path, slurm_config_name: str = 'SLURM_CONFIG') -> None:
    """
    Reads the section with the key-name given by `slurm_config_name` from a config-JSON 
    and creates a SLURM-script and train-script from the settings contained.

    Parameters
    ----------
    config_file : str | Path
        Path to config-JSON.
    slurm_config_name : str, optional
        Name of the section in the config-JSON to read the settings from.\n
        (defaults to `'SLURM_CONFIG'`)
    """
    with open(config_file) as f:
        config: dict[str, Any] = json.load(f)
    config = config[slurm_config_name]
    if 'slurm_options' in config:
        config['slurm_options'] = SlurmOptions(**config['slurm_options'])
    create(**config)


if __name__ == '__main__':
    create_from_config(sys.argv[1], sys.argv[2])
