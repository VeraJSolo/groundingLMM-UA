#!/bin/bash
#SBATCH --job-name=glamm-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu_standard
#SBATCH --account=nlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00

#
#module load cuda/11.7

# Activate environment
source ~/.bashrc
conda activate glamm

# Replace with destination of 
cd "/home/u32/verasjackson/groundingLMM-UA"

# FIX PYTHON PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run inference (replacing with appropraite destination of images and prompts)
srun python capstone_testing/infer.py \
    --hf_model_path "/home/u32/verasjackson/models/GLaMM-FullScope" \ # replace with where you have downloaded GLaMM-FullScope model
    --img_dir "/home/u32/verasjackson/groundingLMM-UA/capstone_testing/image_dataset/test/reference" \
    --output_dir "/home/u32/verasjackson/groundingLMM-UA/capstone_testing/output_inference_colormerging2" \
    --prompt_dir "/home/u32/verasjackson/groundingLMM-UA/capstone_testing/image_dataset/test/text" \
    --prompt_prefix "Please respond with segmentation mask for the TARGET. Take note that when colors overlap, the average is shown (e.g. red and blue will appear as purple, etc)." 

