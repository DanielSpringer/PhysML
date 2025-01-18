# LuttingerWard_from_ML

## Usage
### Setup environment
* install `uv`:
    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv self update
    ```
* install environment:
    ```shell
    uv sync
    ```
* run python in the environment replace `python` with `uv run` in the shell-command. e.g.
    ```shell
    uv run file.py <arguments>
    ```


### Run training
* create config in `/configs/`
* execution on VSC cluster:  
  deploy slurm:  
  ```shell
  sbatch <slurm_filename>.slrm
  ```
* Jupyter-hub on VSC:  
  https://jupyterhub.vsc.ac.at/hub/
  * install dependencies (run in Jupyter-cell):
    ```jupyter
    %pip install -r requirements.txt
    %pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    
    from IPython.display import display, HTML
    display(HTML("<style>.jp-Cell { width: 100% !important;margin-left:00px}</style>"))
    ```
* start tensorboard:
  ```shell
  tensorboard --logdir=saves
  ```


### SLURM
* submit job:
    ```shell
    sbatch <slurm_filename>.slrm
    ```
* show job details:
    ```shell
    scontrol show job <job-id>
    ```
* show job status
    ```shell
    squeue -j <job-id>
    ```
* show job output:
    ```shell
    nano slurm-<job-id>.out
    ```



## ML-framework
Create a new ML-project by deriving classes from the provided base classes and overwrite methods were necessary for customization. 
* It is recommended to put your custom classes in a module-file named after your ML-project (`<project_name>.py`) in the respective component folders (`config`, `load_data`, `models`, `trainer` and `wrapper`).
* In the `__init__.py`-files of the `load_data`-, `models`-, and `wrapper`-module, import everything from your project-specific submodule (`from .<project_name> import *`). This is necessary for the module-loader in the `Config`-class to find your custom classes.
1. Create a config-JSON in containing required parameters. Add a sub-section with SLURM-settings if you want to run the model via SLURM.
2. Create a config-class if additional attributes are required. Tip: You can also create the config-class with default parameter-values first and then save it as a config-JSON by calling the `save`-method.
   - Add a new submodule to `/phys_ml/config/`, containing a class inheriting from `phys_ml.configs.Config`.
   - Add additional attributes.
   - For attributes of type `type`, the fully qualified class-name has to be set.
     - For convenience of accessing/importing types from strings, add each attribute as `_<attribute-name>`, add `<attribute-name>_kwargs` to store arguments for later instantiation and add property-getter, -setter and a `get_<attribute-name>`-method which already returns an imported instance of from the provided class-name.
3. Create a Dataset-class:
   - Add a new submodule to `/phys_ml/load_data/`, containing a class inheriting from `phys_ml.load_data.FilebasedDataset`, which is based on `torch.utils.data.Dataset`.
   - As usual overwrite the `torch.utils.data.Dataset`-methods where required.
4. Create a model class:
   - Add a new submodule to `/phys_ml/models/`, containing a class inheriting from `phys_ml.models.BaseModule`.
   - Overwrite at least the `forward`-method.
5. Create a wrapper-class if custom behaviour is required:
   - Add a new submodule to `/phys_ml/wrapper/`, containing a class inheriting from `phys_ml.wrapper.BaseWrapper`.
   - Overwrite methods where custom behaviour is required.
6. Create a trainer-class if custom behaviour is required:
   - Add a new submodule to `/phys_ml/trainer/`, containing a class inheriting from `phys_ml.trainer.BaseTrainer`.
   - Overwrite methods where custom behaviour is required.

### Usage
There are different options to run the trainer and generate scripts to run it on the cluster:
* run training directly from a Jupyter-notebook by instantiating a Trainer-class in `phys_ml.trainer` and running the `train`-method with the appropriate `train_mode`. By using the `config_kwargs`-argument, you can overwrite parameters in the config for this Trainer-instance (see the `Config`-class for a description of available arguments).
* call `phys_ml.utils.slurm_generate.create` to generate a SLURM-script and a Python train-script to run the training via SLURM. You can also overwrite parameters from the config here by using the `trainer_kwargs`-argument.
* run `phys_ml/util/slurm_generate.py` directly from the command line to generate a SLURM-script and a Python train-script to run the training via SLURM. You can specify the name of the subsection containing the SLURM-settings as an optional argument.  
    ```shell
    python phys_ml/util/slurm_generate.py path/to/config/json <OPTIONAL_NAME_OF_SLURM_SETTINGS_SECTION>
    ```


**Tipps & Tricks:**
- don't copy classes: derive them and overwrite methods where necessary
- don't duplicate code: if a method needs partial customization, try to split it into multiple methods and overwrite only the necessary ones
- variables from one class can be easily shared to other classes by writing it into the config-instance

see `ML_framework_demo.ipynb` for example usage.
