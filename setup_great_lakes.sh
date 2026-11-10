#!/bin/bash

# One-time setup on a Great Lakes login node
# Usage: bash setup_great_lakes.sh

# Load conda
module purge
module load python3.11-anaconda/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the environment
conda create -n alphago python=3.12 -y || { echo "Failed to create conda environment"; exit 1; }
conda activate alphago

# Install required packages
pip install -r requirements.txt

# Check installation
python -c "import torch, numpy; print(f'torch {torch.__version__}, numpy {numpy.__version__}')"
