# Installation

## Currrent local iteration(WIndows)

**note:** originally designed to be able to work on different platforms. I have diverted a little

1. Install cuda
   Using cuda 12.1 for nightly pytorch release
- [cuda 12.1](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
- [cudnn](https://developer.nvidia.com/rdp/cudnn-download)
2. Create enviornmnet

```bash
conda create -n gaze python=3.10
```

3. Install pytorch

**note:** date is 06/02/2023. Nightly and preview versions may have changed
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
```
or
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

4. Install additional dependcies
    
```bash
conda install -c conda-forge black isort mypy pydantic jupyter tensorboard tensorboardX tqdm pytest pylint mkdocs pandas seaborn matplotlib omegaconf zarr sacred mlflow neptune-client scikit-learn-intelex
conda install scikit-learn-intelex
# pip install -r requirements.txt

conda install -c fastai opencv-python-headless
```

## Remove messed up enviornments

```bash
conda remove --name ENV_NAME --all
```

To perform a cleanup of unused packages and cached files in Conda, you can use the conda clean command. There are several options available to clean different types of files. Here are the commonly used options:

1. Clean the package cache:
```bash
conda clean --packages
```
2. Clean the unused cached tarballs:
```bash
conda clean --tarballs
```
3. Clean the index cache:
```bash
conda clean --index-cache
```
4. Clean all unused packages, tarballs, and index cache:
```bash
conda clean --all
```
