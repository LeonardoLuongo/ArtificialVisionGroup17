#!/bin/bash
# Initialize conda for shell interaction
eval "$(conda shell.bash hook)"
conda activate soccer_tracking

# Run the main script
# Adjust the path to your python script if necessary
python main.py