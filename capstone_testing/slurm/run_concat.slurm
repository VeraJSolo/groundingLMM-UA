#!/bin/bash
#SBATCH --job-name=concat-images
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu_standard
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00



# Activate environment
source /opt/ohpc/pub/apps/anaconda/2024.06/etc/profile.d/conda.sh
conda activate glamm

python concat_images.py \
    --input_dir  ~/groundingLMM-UA/dataset_concat/reference \
    --output_dir ~/groundingLMM-UA/dataset_concat/concatenated_images \
    --separator 10
