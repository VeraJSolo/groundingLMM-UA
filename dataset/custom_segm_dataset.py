"""
custom_segm_dataset.py — Custom Dataset for GLaMM Fine-tuning

Expects ONE root folder containing:

    <dataset_root>/
        images/      ← reference .png images  (e.g. GL123_345.png)
        masks/       ← binary mask .png files (e.g. GL123_345.png)
        prompts/     ← text description .txt  (e.g. GL123_345.txt)

The train/val split is handled internally using a fixed random seed so the
same samples are always assigned to the same split across runs.

Usage:
    train_ds = CustomSegmDataset(dataset_root, ..., split="train", val_fraction=0.15)
    val_ds   = CustomSegmDataset(dataset_root, ..., split="val",   val_fraction=0.15)

White pixels (>127) in the mask = foreground target region.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from tools.utils import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)

# Instruction templates — randomly sampled during training for variety,
# fixed to index 0 during validation for reproducibility.
# All templates must elicit [SEG] tokens from the model.
SEG_INSTRUCTIONS = [
    "Please segment the described target: {description}",
    "Segment the following object in the image: {description}",
    "Provide a segmentation mask for: {description}",
    "Please respond with a segmentation mask for the described target: {description}",
]


class CustomSegmDataset(Dataset):
    """
    Custom segmentation dataset with automatic train/val split.

    Args:
        dataset_root        (str)  : Root folder containing images/, masks/, prompts/.
        tokenizer                  : HuggingFace tokenizer.
        global_image_encoder (str) : CLIP model name for CLIPImageProcessor.
        split               (str)  : "train" or "val".
        val_fraction        (float): Fraction of data reserved for validation (default 0.15).
        split_seed          (int)  : Random seed for reproducible splitting (default 42).
        image_size          (int)  : SAM grounding encoder input size (default 1024).
        precision           (str)  : "bf16" or "fp16".
        prompt_prefix       (str)  : If set, overrides random instruction templates.
        use_mm_start_end    (bool) : Wrap image token with <im_start>/<im_end>.
    """

    def __init__(
        self,
        dataset_root,
        tokenizer,
        global_image_encoder,
        split="train",
        val_fraction=0.15,
        split_seed=42,
        max_samples=None,
        image_size=1024,
        precision="bf16",
        prompt_prefix=None,
        use_mm_start_end=True,
        # Silently accepted kwargs for compatibility with common_ds_args
        dataset_dir=None,
        epoch_samples=None,
        num_classes_per_sample=None,
        random_sampling=None,
        validation=None,   # ignored — use split= instead
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
        assert 0.0 < val_fraction < 1.0, "val_fraction must be between 0 and 1"

        self.dataset_root    = dataset_root
        self.image_dir       = os.path.join(dataset_root, "reference")
        self.mask_dir        = os.path.join(dataset_root, "target")
        self.prompt_dir      = os.path.join(dataset_root, "text")
        self.split           = split
        self.val_fraction    = val_fraction
        self.split_seed      = split_seed
        self.max_samples     = max_samples
        self.tokenizer       = tokenizer
        self.image_size      = image_size
        self.precision       = precision
        self.prompt_prefix   = prompt_prefix
        self.use_mm_start_end = use_mm_start_end
        self.is_validation   = (split == "val")

        # Validate directories
        for d, name in [(self.image_dir,  "images"),
                        (self.mask_dir,   "masks"),
                        (self.prompt_dir, "prompts")]:
            if not os.path.isdir(d):
                raise ValueError(f"CustomSegmDataset: '{name}' folder not found at {d}")

        # Build full matched sample list, subsample, then split
        all_samples = self._build_sample_list()
        all_samples = self._subsample(all_samples)
        self.samples = self._split_samples(all_samples)

        print(f"[CustomSegmDataset] split={split:5s} | "
              f"{len(self.samples)} samples "
              f"(pool={len(all_samples)}, val_fraction={val_fraction}, seed={split_seed})")

        # Image processors
        self.clip_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.sam_transform  = ResizeLongestSide(image_size)
        self.torch_dtype    = torch.bfloat16 if precision == "bf16" else torch.float16

        # SAM pixel normalisation constants
        self.pixel_mean = torch.tensor([123.675, 116.28,  103.53]).view(3, 1, 1)
        self.pixel_std  = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------

    def _build_sample_list(self):
        """Return sorted list of stems with all three files present."""
        image_stems  = {os.path.splitext(f)[0] for f in os.listdir(self.image_dir)
                        if f.lower().endswith(".png")}
        mask_stems   = {os.path.splitext(f)[0] for f in os.listdir(self.mask_dir)
                        if f.lower().endswith(".png")}
        prompt_stems = {os.path.splitext(f)[0] for f in os.listdir(self.prompt_dir)
                        if f.lower().endswith(".txt")}

        complete = sorted(image_stems & mask_stems & prompt_stems)

        missing_mask   = image_stems - mask_stems
        missing_prompt = image_stems - prompt_stems
        if missing_mask:
            print(f"  [WARNING] {len(missing_mask)} image(s) missing mask — skipped: "
                  f"{sorted(missing_mask)[:5]}{'...' if len(missing_mask) > 5 else ''}")
        if missing_prompt:
            print(f"  [WARNING] {len(missing_prompt)} image(s) missing prompt — skipped: "
                  f"{sorted(missing_prompt)[:5]}{'...' if len(missing_prompt) > 5 else ''}")

        if len(complete) == 0:
            raise RuntimeError(f"No complete triplets found in {self.dataset_root}")

        return complete

    def _subsample(self, all_samples):
        """
        Randomly select up to max_samples from the full pool before splitting.
        Uses split_seed for reproducibility — same seed = same subset every run.
        If max_samples is None or >= pool size, returns the full list unchanged.
        """
        if self.max_samples is None or self.max_samples >= len(all_samples):
            return all_samples
        rng = np.random.default_rng(self.split_seed)
        indices = rng.choice(len(all_samples), size=self.max_samples, replace=False)
        subset = [all_samples[i] for i in sorted(indices)]
        print(f"  [INFO] Subsampled {self.max_samples} from {len(all_samples)} available samples.")
        return subset

    def _split_samples(self, all_samples):
        """
        Deterministically split samples into train and val using split_seed.
        The same seed always produces the same split, regardless of run order.
        """
        rng = np.random.default_rng(self.split_seed)
        indices = np.arange(len(all_samples))
        rng.shuffle(indices)                            # shuffle with fixed seed

        n_val   = max(1, int(len(all_samples) * self.val_fraction))
        n_train = len(all_samples) - n_val

        val_indices   = indices[:n_val]
        train_indices = indices[n_val:]

        if self.split == "train":
            return [all_samples[i] for i in sorted(train_indices)]
        else:
            return [all_samples[i] for i in sorted(val_indices)]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_image(self, stem):
        path = os.path.join(self.image_dir, stem + ".png")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, stem):
        """Load binary mask. White pixels (>127) = foreground."""
        path = os.path.join(self.mask_dir, stem + ".png")
        mask = np.array(Image.open(path).convert("L"))
        return mask > 127   # bool (H, W)

    def _load_prompt(self, stem):
        path = os.path.join(self.prompt_dir, stem + ".txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _build_instruction(self, description):
        """
        Prepend a segmentation instruction to the .txt description.
        Training: randomly sample from SEG_INSTRUCTIONS for variety.
        Validation: always use index 0 for reproducible evaluation.
        """
        if self.prompt_prefix is not None:
            template = self.prompt_prefix + "{description}"
        elif self.is_validation:
            template = SEG_INSTRUCTIONS[0]
        else:
            template = SEG_INSTRUCTIONS[np.random.randint(len(SEG_INSTRUCTIONS))]
        return template.format(description=description)

    def _preprocess_grounding_image(self, image_np):
        """Resize + normalize + pad for SAM grounding encoder."""
        image = self.sam_transform.apply_image(image_np)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        h, w = tensor.shape[-2:]
        padded = torch.zeros(3, self.image_size, self.image_size)
        padded[:, :h, :w] = tensor
        return padded.to(self.torch_dtype)

    def _build_conversation_tokens(self, instruction):
        """Wrap instruction in LLaVA conversation format and tokenize."""
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        begin_str = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"
        prompt = begin_str + instruction
        if self.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt_str = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_str, self.tokenizer, return_tensors="pt")
        return input_ids, conv

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]

        image_np    = self._load_image(stem)
        gt_mask     = self._load_mask(stem)
        description = self._load_prompt(stem)
        instruction = self._build_instruction(description)

        # CLIP global encoder input
        global_enc_image = self.clip_processor.preprocess(
            image_np, return_tensors="pt"
        )["pixel_values"][0]                              # (3, 336, 336)

        # SAM grounding encoder input
        grounding_enc_image = self._preprocess_grounding_image(image_np)  # (3, 1024, 1024)

        # Sizes needed for SAM mask upsampling
        original_size = torch.tensor(image_np.shape[:2])
        resize_size   = torch.tensor(self.sam_transform.apply_image(image_np).shape[:2])

        # Tokenize
        input_ids, conv = self._build_conversation_tokens(instruction)

        # Labels — mask instruction tokens with -100 so CE loss only
        # applies to the model's response (standard LLaVA training practice)
        labels = input_ids.clone()
        sep_ids = self.tokenizer.encode(
            conv.sep + conv.roles[1] + ":", add_special_tokens=False
        )
        labels[:len(sep_ids)] = -100

        # GT mask tensor: (1, H, W)
        gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0)

        return {
            "global_enc_images":    global_enc_image,
            "grounding_enc_images": grounding_enc_image,
            "input_ids":            input_ids,
            "labels":               labels,
            "attention_masks":      input_ids.ne(self.tokenizer.pad_token_id),
            "masks_list":           [gt_mask_tensor],
            "label_list":           [torch.tensor([1])],
            "image_sizes":          [original_size],
            "resize_list":          [resize_size],
            "inference":            False,
            "image_paths":          os.path.join(self.image_dir, stem + ".png"),
            "conversations":        [instruction],
        }
