import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_size=1024):
        self.root = root_dir
        self.tokenizer = tokenizer

        self.image_dir = os.path.join("/home/u32/verasjackson/groundingLMM-UA/test_files/shortlisted_testing_dataset/reference", "images")
        self.prompt_dir = os.path.join("/home/u32/verasjackson/groundingLMM-UA/test_files/shortlisted_testing_dataset/text", "prompts")
        self.target_dir = os.path.join("/home/u32/verasjackson/groundingLMM-UA/test_files/shortlisted_testing_dataset/target", "targets")

        # get list of IDs (filenames without extension)
        self.ids = [f.split(".")[0] for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        # --- Load image ---
        image_path = os.path.join(self.image_dir, sample_id + ".png")
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        # --- Load prompt ---
        prompt_path = os.path.join(self.prompt_dir, sample_id + ".txt")
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()

        # --- Load target mask ---
        target_path = os.path.join(self.target_dir, sample_id + ".png")
        mask = Image.open(target_path).convert("L")
        mask = np.array(mask)
        mask = torch.tensor(mask).unsqueeze(0)  # [1, H, W]

        # --- Tokenize text ---
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = inputs["input_ids"][0]

        return {
            "global_enc_images": image,
            "grounding_enc_images": image,
            "input_ids": input_ids,
            "labels": input_ids,  # can refine later
            "gt_masks": mask
        }
