#!/usr/bin/env python3
"""Run GLaMM on one image with a custom text prompt (no Gradio / no Slurm required)."""
import argparse
import json
import os
import re
import sys

import bleach
import cv2
import torch
from transformers import AutoTokenizer, CLIPImageProcessor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.utils import coco_encode_rle, grounding_image_ecoder_preprocess, mask_to_rle_pytorch
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def parse_args():
    p = argparse.ArgumentParser(description="GLaMM single-image inference")
    p.add_argument("--hf_model_path", required=True, help="HF hub id or local folder (e.g. MBZUAI/GLaMM-FullScope)")
    p.add_argument("--image", required=True, help="Path to one image file")
    p.add_argument(
        "--prompt",
        default="Describe this image in detail. For each main object or region, include a segmentation mask in your answer.",
        help="User instruction (GCG-style grounded captioning works best with GLaMM-GCG)",
    )
    p.add_argument("--output_json", default=None, help="Optional path to save JSON (caption + mask RLEs)")
    p.add_argument("--image_size", default=1024, type=int)
    p.add_argument("--model_max_length", default=512, type=int)
    p.add_argument("--max_tokens_new", default=512, type=int)
    p.add_argument("--use_mm_start_end", action="store_true", default=True)
    p.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    return p.parse_args()


def run_inference(args, tokenizer, model, clip_image_processor, transform, instructions, image_path):
    instructions = bleach.clean(instructions)
    instructions = instructions.replace("&lt;", "<").replace("&gt;", ">")

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(image_path)
    if image_np is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    image_clip = image_clip.bfloat16()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = grounding_image_ecoder_preprocess(
        torch.from_numpy(image).permute(2, 0, 1).contiguous()
    ).unsqueeze(0).cuda()
    image = image.bfloat16()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip, image, input_ids, resize_list, original_size_list, max_tokens_new=args.max_tokens_new, bboxes=None
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r"<.*?>", "", text_output)
    pattern = re.compile(r"<p>(.*?)</p>")
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]
    cleaned_str = cleaned_str.replace("[SEG]", "")
    cleaned_str = " ".join(cleaned_str.split()).strip("'").strip()

    return cleaned_str, pred_masks, phrases, text_output


def main():
    args = parse_args()
    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        sys.exit(f"Not a file: {image_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path, cache_dir=None, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16
    model = GLaMMForCausalLM.from_pretrained(
        args.hf_model_path, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, torch_dtype=torch_dtype
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model = model.bfloat16().cuda()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    caption, pred_masks, phrases, raw = run_inference(
        args, tokenizer, model, clip_image_processor, transform, args.prompt, image_path
    )

    print(caption)
    if args.output_json:
        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]
        out = {
            "image": image_path,
            "prompt": args.prompt,
            "caption": caption,
            "phrases": phrases,
            "pred_masks": rle_masks,
            "raw_assistant": raw,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
