#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml --force
conda activate apec
cat requirements.txt | xargs -n 1 -L 1 pip install --no-cache --no-deps
