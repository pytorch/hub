#!/bin/bash
set -ex

CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
bash "$filename" -b -u

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base
conda install -y python=3.8
conda update -y conda
