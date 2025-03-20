# ATTACC: Attention Based Accident Anticipation
## A Monocular Depth Enhanced Approach




## Environment Setup


Follow the steps below to set up the project environment exactly as configured.

### 1. Create and Activate Conda Environment
```sh
conda create -n attacc python=3.10
conda activate attacc
```

**Note:** PyTorch announced that version 2.5 will be the last release published to the `pytorch` channel on Anaconda.

### 2. Install PyTorch

#### macOS
Run the following command to install PyTorch:
```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
```

#### Linux and Windows
Depending on your CUDA version, use one of the following commands:

**CUDA 11.8:**
```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**CUDA 12.1:**
```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CUDA 12.4:**
```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**CPU Only:**
```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```

### 3. Install Poetry
Install Poetry for package management:
```sh
pip install poetry
```

### 4. Install Project Dependencies
Once Poetry is installed, use the following command to install all dependencies from the `poetry.lock` file:
```sh
poetry install
```

### 5. Install `timm`
Since installing `timm` via Poetry unnecessarily updates PyTorch to version 2.6, install it using `pip` instead:
```sh
pip install timm==0.6.12
```

