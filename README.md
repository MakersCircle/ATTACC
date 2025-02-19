# ATTACC: Attention Based Accident Anticipation
## A Monocular Depth Enhanced Approach




### Environment Setup
1. conda create -n attacc python=3.10
2. conda activate attacc

   Note: Pytorch announced that 2.5 will be the last release of PyTorch that will be published to the pytorch channel on Anaconda.
3. conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
OSX
conda
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
Linux and Windows
CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
CPU Only
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
4. pip install poetry
5. poetry init
6. poetry add numpy
7. poetry add opencv-python
8. poetry add tqdm
9. 