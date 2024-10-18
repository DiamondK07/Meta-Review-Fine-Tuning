#!/bin/bash

# Source Conda script to make 'conda activate' available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate env  # Replace 'myenv' with your environment name

# Navigate to the directory where FT.py is located
cd ~/diamond  # Navigate to the 'diamond' directory

# Run the Python script and log the output
python summarize.py &> summarize.log

