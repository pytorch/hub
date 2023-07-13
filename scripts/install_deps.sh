#!/bin/bash
set -ex

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
conda install -y pytest

# Dependencies required to load models
conda install -y regex pillow tqdm boto3 requests numpy h5py scipy matplotlib unidecode ipython pyyaml
conda install -y -c conda-forge librosa inflect

pip install -q fastBPE sacremoses sentencepiece subword_nmt editdistance
pip install -q visdom filelock tokenizers packaging pandas
pip install -q mistune==2.0.5
pip install -q omegaconf timm seaborn importlib_metadata huggingface_hub
pip install -q hydra-core opencv-python fvcore
pip install -q --upgrade google-api-python-client
pip install pytorchvideo
pip install -q prefetch_generator  # yolop
pip install -q pretrainedmodels efficientnet_pytorch webcolors # hybridnets
