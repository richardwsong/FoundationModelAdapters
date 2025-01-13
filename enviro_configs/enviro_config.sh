#!/bin/bash

set -e

echo "Creating Conda environment 'brain_clip_adapter' with Python 3.8.18..."
conda create --name brain_clip_adapter python=3.8.18 -y

echo "Activating Conda environment 'brain_clip_adapter'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate brain_clip_adapter

echo "Installing dependencies..."
conda install -y numpy
conda install -y pandas
conda install -y conda-forge::nibabel
conda install -y conda-forge::nilearn
conda install -y conda-forge::mne
conda install -y pytorch::pytorch
conda install -y conda-forge::tensorboard
conda install -y conda-forge::tensorboardx
pip install shap
conda install -y conda-forge::opencv
conda install -y anaconda::colorcet
pip install umap-learn
pip install plotly
pip install torchvision
pip install git+https://github.com/openai/CLIP.git

echo "Conda environment 'brain_clip_adapter' setup is complete!"