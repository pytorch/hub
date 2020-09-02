#!/bin/bash
set -e

. ~/miniconda3/etc/profile.d/conda.sh
conda activate base
conda install -y pytorch torchtext torchvision -c pytorch-nightly

