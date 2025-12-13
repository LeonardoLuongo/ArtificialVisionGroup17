#!/bin/bash
echo "Building Conda environment 'soccer_tracking'..."
conda env create -f conda/build/requirements.yaml
echo "Build complete."