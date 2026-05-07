#!/bin/bash
#SBATCH --job-name=glamm-performance
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

#
#module load cuda/11.7

# Activate environment
source ~/.bashrc
conda activate glamm

# Go to repo
cd "/home/u32/verasjackson/groundingLMM-UA"

# FIX PYTHON PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run evluation
python evaluation.py \
    --json_dir   "/home/u32/verasjackson/groundingLMM-UA/output_inference_colormerging" \
    --gt_dir     "/home/u32/verasjackson/groundingLMM-UA/image_dataset/test/target" \
    --output_dir "/home/u32/verasjackson/groundingLMM-UA/performance_result"
