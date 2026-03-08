# Installation Instruction
This document provides instructions of how to properly install this codebase. We recommend using a conda environment to simplify setup.
## Setup a Conda Environment

This repo requires [pytorch3d](https://github.com/facebookresearch/pytorch3d), which can be installed as follows.
```
conda create -n robosyn python=3.8
conda init bash
conda activate robosyn
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pytorch3d
```

## IsaacGym
Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. We provide the bash commands we used.
```
wget -O IsaacGym_Preview_4_Package.tar.gz https://developer.nvidia.com/isaac-gym-preview-4
pip install scipy imageio ninja
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e . --no-deps
```

## Other dependencies
Install the rest dependencies of this repo via pip:
```
pip install hydra-core gym ray open3d numpy==1.23.5 tensorboardX tensorboard wandb
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

You are all set now! Follow the main instructions to continue the exploration journey.