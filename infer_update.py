import os
import re
import cv2
import json
import bleach
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
from eval.utils import *
from eval.ddp import *
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - GCG")
    parser.add_argument("--hf_model_path", required=True,
                        help="The model path in huggingface format.")
    parser.add_argument("--img_dir", required=False,
                        default="./data/GranDf/GranDf_HA_images/val_test",
                        help="The directory containing images to run inference.")

    # --- Prompt arguments (mutually exclusive) ---
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt_dir", type=str, default=None,
                              help="Directory of .txt files. Each file must share the "
                                   "same stem as its corresponding .png "
                                   "(e.g. GL123_345.txt for GL123_345.png).")
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="Single prompt string applied to every image. "
                                   "Ignored when --prompt_dir is set.")
    parser.add_argument("--fallback_prompt", type=str,
                        default="Could you please give me a detailed description of "
                                "the image? Please respond with interleaved "
                                "segmentation masks for the corresponding parts of "
                                "the answer.",
                        help="Prompt used when --prompt_dir is set but no matching "
                             ".txt file is found for an image.")
    parser.add_argument("--prompt_prefix", type=str,
                        default="Please segment the described target: ",
                        help="Text prepended to every .txt file content when using "
                             "--prompt_dir. Set to '' to use the .txt content as-is. "
                             "Default: 'Please segment the described target: '")
    # ---------------------------------------------

    parser.add_argument("--output_dir", required=True,
                        help="The directory to store the response in json format.")
    parser.add_argument("--image_size", default=1024, type=int,
                        help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str,
                        choices=["llava_v1", "llava_llama_2"])

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://",
                        help="url used to set up distributed training")
    return parser.parse_args()


def load_prompt_for_image(image_id, prompt_dir, fallback_prompt): # ADDITION
    """
    Given an image_id (e.g. 'GL123_345.png'), look for a matching
    .txt file in prompt_dir (e.g. prompt_dir/GL123_345.txt).

    Returns the file contents (stripped) if found, otherwise fallback_prompt.
    Logs a warning for missing files so you can audit mismatches easily.
    """
    stem = os.path.splitext(image_id)[0]          # 'GL123_345.png' → 'GL123_345'
    prompt_path = os.path.join(prompt_dir, stem + ".txt")

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
        else:
            print(f"[WARNING] Prompt file is empty: {prompt_path} — using fallback.")
            return fallback_prompt
    else:
        print(f"[WARNING] No prompt file found for '{image_id}' "
              f"(expected: {prompt_path}) — using fallback prompt.")
        return fallback_prompt


def inference(instructions, image_path):
    # Filter out special chars
    instructions = bleach.clean(instructions)
    instructions = instructions.replace("&lt;", "<").replace("&gt;", ">")

    # Prepare prompt for model inference
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

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0).cuda()
    )
    image_clip = image_clip.bfloat16()

    # Preprocess the image (Grounding image encoder - SAM)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous()
        ).unsqueeze(0).cuda()
    )
    image = image.bfloat16()

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None  # No box/region input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(
        image_clip, image, input_ids, resize_list, original_size_list,
        max_tokens_new=512, bboxes=bboxes
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    cleaned_str = re.sub(r"<.*?>", "", text_output)
    pattern = re.compile(r"<p>(.*?)<\/p>")
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]
    cleaned_str = cleaned_str.replace("[SEG]", "")
    cleaned_str = " ".join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    return cleaned_str, pred_masks, phrases


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]
    return image_id, image_path


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)

    # Validate prompt arguments (ADDITION)
    if args.prompt_dir is not None and not os.path.isdir(args.prompt_dir):
        raise ValueError(f"--prompt_dir does not exist or is not a directory: {args.prompt_dir}")

    # Determine prompt mode once, clearly (ADDITION)
    using_prompt_dir = args.prompt_dir is not None
    if using_prompt_dir:
        print(f"[INFO] Prompt mode: per-image .txt files from '{args.prompt_dir}'")
        print(f"[INFO] Prompt prefix: \"{args.prompt_prefix}\"")
        print(f"[INFO] Fallback prompt: \"{args.fallback_prompt}\"")
    else:
        global_prompt = args.prompt if args.prompt is not None else args.fallback_prompt
        print(f"[INFO] Prompt mode: single global prompt — \"{global_prompt}\"")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path, cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.bfloat16
    kwargs = {"torch_dtype": torch_dtype}
    model = GLaMMForCausalLM.from_pretrained(
        args.hf_model_path, low_cpu_mem_usage=True,
        seg_token_idx=seg_token_idx, **kwargs
    )

    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize Global Image Encoder (CLIP)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer model to GPU
    model = model.bfloat16().cuda() # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize CLIP image processor and SAM transform
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create DDP dataset and dataloader
    dataset = GCGEvalDDP(args.img_dir)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
        sampler=distributed_sampler, collate_fn=custom_collate_fn
    )

    # Track missing prompt files for a final summary (ADDITION)
    missing_prompts = []

    # Iterate over images, run inference, save results
    for (image_id, image_path) in tqdm(dataloader):
        image_id, image_path = image_id[0], image_path[0]
        output_path = os.path.join(args.output_dir, image_id[:-4] + ".json")

        # --- Resolve the prompt for this specific image --- (ADDITION)
        if using_prompt_dir:
            stem = os.path.splitext(image_id)[0]
            prompt_path = os.path.join(args.prompt_dir, stem + ".txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read().strip()
                if txt_content:
                    # Prepend segmentation instruction to the .txt description
                    # e.g. "Please segment the described target: a red bicycle near the fence"
                    instruction = f"{args.prompt_prefix}{txt_content}"
                else:
                    print(f"[WARNING] Empty prompt file: {prompt_path} — using fallback.")
                    instruction = args.fallback_prompt
            else:
                missing_prompts.append(image_id)
                instruction = args.fallback_prompt
        else:
            instruction = global_prompt
        # --------------------------------------------------

        result_caption, pred_masks, phrases = inference(instruction, image_path)

        # Convert predicted masks to RLE format
        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]

        # Save results
        result_dict = {
            "image_id": image_id[:-4],
            "prompt_used": instruction,          # logged so you can audit later
            "caption": result_caption,
            "phrases": phrases,
            "pred_masks": rle_masks,
        }
        
        # Save the inference results
        with open(output_path, "w") as f:
            json.dump(result_dict, f)

    # Final audit summary (ADDITION)
    if missing_prompts:
        print(f"\n[AUDIT] {len(missing_prompts)} image(s) had no matching .txt prompt "
              f"and used the fallback:")
        for name in missing_prompts:
            print(f"  - {name}")
    else:
        if using_prompt_dir:
            print("\n[AUDIT] All images had a matching .txt prompt file.")