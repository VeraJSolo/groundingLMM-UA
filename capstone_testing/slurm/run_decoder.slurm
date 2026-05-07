#!/bin/bash
#SBATCH --job-name=decode-mask
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

python decode_mask.py \
	--json_dir "/home/u32/verasjackson/groundingLMM-UA" \
	--img_dir test_files/shortlisted_testing_dataset/reference \
	--output_dir "/home/u32/verasjackson/groundingLMM-UA"

