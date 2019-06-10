#!/bin/bash
set -e

# Install basics
sudo apt-get install vim

# Install miniconda
CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Install cpu version of pytorch and torchvision
conda install -y pytorch torchvision -c pytorch

# Dependencies required to load models
conda install -y regex pillow tqdm boto3 requests numpy h5py scipy matplotlib

pip install -q visdom
