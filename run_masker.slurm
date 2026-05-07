#!/bin/bash
#SBATCH --job-name=mask-image
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

python maskimage.py \
        --json_dir "/home/u32/verasjackson/groundingLMM-UA/output_inference_colormerging" \
        --output_dir "/home/u32/verasjackson/groundingLMM-UA/output_inference_colormerging/mask"
