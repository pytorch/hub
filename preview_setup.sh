#!/bin/bash
git clone --recursive https://github.com/pytorch/pytorch.github.io.git -b site
cp *.md pytorch.github.io/_hub
cp images/* pytorch.github.io/assets/images/

