import argparse
import contextlib
import functools
import gc
import logging
import math
import os
import random
import shutil
from collections import Counter
from datetime import timedelta

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, concatenate_datasets
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import load_image, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft, \
    convert_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, GroundingDinoProcessor, GroundingDinoModel

from grounded_sam.grounding_dino import GroundingDino
from models.unet_2d_condition import ImageVariationUNet2DConditionModel as UNet2DConditionModel
from pipelines.pipeline_stable_xl_image_variation import StableDiffusionXLImageVariationPipeline

logger = get_logger(__name__)


def log_validation(unet, args, accelerator, weight_dtype, step, image_encoder, image_encoder2, class_labels, is_final_validation=False):
    logger.info("Running validation... ")
    if args.grounding_dino_hidden_states:
        gd_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        gd_model = GroundingDinoModel.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        gd_model.requires_grad_(False)
        gd_model.to(accelerator.device)
    else:
        gd_processor = None
        gd_model = None

    if not is_final_validation:
        unet = accelerator.unwrap_model(unet)
        if args.train_image_encoder:
            image_encoder = accelerator.unwrap_model(image_encoder)
            image_encoder2 = accelerator.unwrap_model(image_encoder2)
            pipeline = StableDiffusionXLImageVariationPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                image_encoder=image_encoder,
                image_encoder2=image_encoder2,
                gd_processor=gd_processor,
                gd_model=gd_model,
                torch_dtype=weight_dtype,
                local_files_only=True
            )
        else:
            pipeline = StableDiffusionXLImageVariationPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                gd_processor=gd_processor,
                gd_model=gd_model,
                torch_dtype=weight_dtype,
                local_files_only=True
            )
    else:
        pipeline = StableDiffusionXLImageVariationPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            gd_processor=gd_processor,
            gd_model=gd_model,
            torch_dtype=weight_dtype,
            local_files_only=True
        )
        pipeline.load_lora_weights(args.output_dir)
        if args.class_condition:
            from models.class_embed_adapter import ClassEmbedAdapter
            adapter = ClassEmbedAdapter(args, class_labels)
            adapter = adapter.to(accelerator.device, dtype=weight_dtype)
            pipeline.set_class_embedding_adapter(adapter)
            pipeline.load_class_embedding_adapter(os.path.join(args.output_dir, "unet_class_embedding.pt"))

    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.dataset == "COCO":
        eval_img_list = ["000000000009.jpg", "000000000025.jpg", "000000000030.jpg", "000000000139.jpg", "000000000285.jpg", "000000000632.jpg"]
        if args.class_condition:
            class_labels = [["orange", "bowl", "broccoli"], ["giraffe"], ["potted plant", "vase"], [], [], []]
        else:
            class_labels = [None] * len(eval_img_list)

        if args.grounding_dino_hidden_states:
            class_label = ["orange. bowl. broccoli.", "giraffe.", "potted plant. vase.", "", "", ""]
        else:
            class_label = [None] * len(eval_img_list)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    eval_img = [load_image(os.path.join("src", img_path)) for img_path in eval_img_list]
    output_images_list = []
    for idx in range(len(eval_img_list)):
        with inference_ctx:
            image = pipeline(
                eval_img[idx], class_label=class_label[idx], class_labels=[class_labels[idx]], generator=generator,
                height=args.resolution, width=args.resolution
            ).images[0]

        output_images_list.append(image)

    tracker_key = "test" if is_final_validation else f"validation-steps{step}"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            table_data = []

            for idx in range(len(eval_img_list)):
                input_image = eval_img[idx]
                output_image = output_images_list[idx]
                table_data.append([
                    wandb.Image(input_image, caption="Input image"),
                    wandb.Image(output_image, caption="Output image"),
                ])

            tracker.log({tracker_key: wandb.Table(data=table_data, columns=["Input", "Output"])})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL Image Variation Training")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='COCO',
    )
    parser.add_argument(
        "--use_content",
        action="store_true",
        help="Whether to use content dataset."
    )
    parser.add_argument(
        "--content_type",
        type=str,
        default="gt",
        choices=["gt", "random_crop", "bbox_jitter"],
        help="The type of content to use."
    )
    parser.add_argument(
        "--max_content_length",
        type=int,
        default=9,
        help="The maximum number of content images."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the wandb run."
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="image-variation-sdxl",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if re-balancing the loss. Recommended value is 5.0. "
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--train_image_encoder",
        action="store_true",
        help="Whether to train the image encoder."
    )
    parser.add_argument(
        "--grounding_dino_hidden_states",
        action="store_true",
        help="Whether to use hidden states from DINO."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--class_condition",
        action="store_true",
        help="Whether to use class condition."
    )
    parser.add_argument(
        "--counting_loss",
        action="store_true",
        help="Whether to use grounded sam loss."
    )
    parser.add_argument(
        "--counting_loss_threshold",
        type=float,
        default=0.2,
        help="Threshold for counting loss."
    )
    parser.add_argument(
        "--counting_loss_scale",
        type=float,
        default=1.0,
        help="Scale for counting loss."
    )
    parser.add_argument(
        "--counting_loss_steps",
        type=int,
        default=0,
        help="Number of steps to start train the counting loss."
    )
    parser.add_argument(
        "--denoise_type",
        type=str,
        default="step",
        choices=["step"], # "minus"
        help="The type of denoising to use."
    )
    args = parser.parse_args()

    return args


def make_train_dataset(args, accelerator):
    if args.dataset == "COCO":
        file_name = 'coco.py'
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    dataset = load_dataset(f"dataset/{file_name}", split="train", trust_remote_code=True)
    image_column = "image"
    caption_column = "captions"
    class_labels = dataset.features['instances'][0]['label']
    assert isinstance(dataset, datasets.Dataset), "Dataset should be of type `datasets.Dataset`"

    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]

        if args.use_content:
            if args.content_type == "gt":
                from utils.training_utils import crop_content
                contents = crop_content(examples)
            elif args.content_type == "random_crop":
                from utils.training_utils import random_crop_content
                contents = random_crop_content(examples)
            elif args.content_type == "bbox_jitter":
                from utils.training_utils import bbox_jitter_content
                contents = bbox_jitter_content(examples)
            else:
                raise ValueError(f"Unknown content type {args.content_type}")
            examples["contents"] = contents

        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images

        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    return train_dataset, image_column, caption_column, class_labels


def encode_image(batch, image_processors, image_encoders, vae, image_column, args):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        image_latents = vae.encode(pixel_values).latent_dist.sample()
    img_latents = image_latents * vae.config.scaling_factor

    image_embeds_list = []
    with torch.no_grad():
        for image_processor, image_encoder in zip(image_processors, image_encoders):
            cond_images = image_processor.preprocess(batch[image_column], return_tensors='pt').data['pixel_values']
            cond_images = torch.stack(list(cond_images))
            cond_images = cond_images.to(memory_format=torch.contiguous_format).float()
            cond_images = cond_images.to(image_encoder.device, dtype=image_encoder.dtype)

            image_embeds = image_encoder(cond_images, output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_image_embeds = image_embeds[0]
            image_embeds = image_embeds.image_embeds.unsqueeze(1)
            bs_embed, seq_len, _ = image_embeds.shape
            image_embeds = image_embeds.view(bs_embed, seq_len, -1)
            image_embeds_list.append(image_embeds)

    image_embeds = torch.cat(image_embeds_list, dim=-1)
    pooled_image_embeds = pooled_image_embeds.view(bs_embed, -1)

    batch["image_embeds"] = image_embeds
    batch["pooled_image_embeds"] = pooled_image_embeds
    batch["img_latents"] = img_latents

    return batch

def encode_image_with_content(batch, image_processors, image_encoders, vae, image_column, args):
    images = batch.pop("pixel_values")
    contents = batch.pop("contents")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        image_latents = vae.encode(pixel_values).latent_dist.sample()
    img_latents = image_latents * vae.config.scaling_factor

    image_embeds_list = []
    with torch.no_grad():
        for image_processor, image_encoder in zip(image_processors, image_encoders):
            cond_images = image_processor.preprocess(batch[image_column], return_tensors='pt').data['pixel_values']
            cond_images = torch.stack(list(cond_images))
            cond_images = cond_images.to(memory_format=torch.contiguous_format).float()
            cond_images = cond_images.to(image_encoder.device, dtype=image_encoder.dtype)

            image_embeds = image_encoder(cond_images).image_embeds

            cont_embed_list = []
            for cont in contents:
                if not cont:
                    cont_embed_list.append(torch.zeros(args.max_content_length, image_embeds.shape[1], device=image_encoder.device))
                    continue
                cont = image_processor.preprocess(cont, return_tensors='pt').data['pixel_values']
                cont = torch.stack(list(cont))
                cont = cont.to(memory_format=torch.contiguous_format).float()
                cont = cont.to(image_encoder.device, dtype=image_encoder.dtype)
                cont_embeds = image_encoder(cont).image_embeds
                cont_embed_list.append(cont_embeds)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_image_embeds = image_embeds

            # We add content embeddings to image embeddings for the final image embeddings
            padding_list = []
            for img_emb, cont_emb in zip(list(image_embeds), cont_embed_list):
                if cont_emb.shape[0] <= args.max_content_length:
                    zeros_emb = torch.zeros(args.max_content_length - cont_emb.shape[0], cont_emb.shape[1], device=cont_emb.device)
                    cont_emb = torch.cat([cont_emb, zeros_emb], dim=0)
                else:
                    cont_emb = cont_emb[torch.randperm(cont_emb.shape[0])]
                padding_list.append(torch.cat([img_emb.unsqueeze(0), cont_emb[:args.max_content_length]], dim=0))
            image_embeds = torch.stack(padding_list)
            bs_embed, seq_len, _ = image_embeds.shape
            image_embeds = image_embeds.view(bs_embed, seq_len, -1)
            image_embeds_list.append(image_embeds)

    image_embeds = torch.cat(image_embeds_list, dim=-1)
    pooled_image_embeds = pooled_image_embeds.view(bs_embed, -1)

    batch["image_embeds"] = image_embeds
    batch["pooled_image_embeds"] = pooled_image_embeds
    batch["img_latents"] = img_latents

    return batch


def encode_image_with_grounding_dino(batch, processor, grounding_dino, class_labels):
    images = [image.convert("RGB") for image in batch["image"]]
    labels = []
    hidden_states = []
    for instances in batch["instances"]:
        image_labels = []
        for instance in instances:
            label_name = class_labels.int2str(instance["label"])
            image_labels.append(label_name)
        name_count = Counter(image_labels)
        text_prompt = ". ".join(name_count.keys()) + "." if len(name_count) != 0 else ""
        labels.append(text_prompt)
    for image, text in zip(images, labels):
        inputs = processor(images=image, text=text, return_tensors="pt").to(grounding_dino.device)
        with torch.no_grad():
            hidden_state = grounding_dino(**inputs).last_hidden_state[:, -5:, :].flatten()
        hidden_states.append(hidden_state)
    hidden_state = torch.stack(hidden_states)

    return {"hidden_states": hidden_state}

def encode_image_without_batch(image_processors, image_encoders, images):
    image_embeds_list = []

    for image_processor, image_encoder in zip(image_processors, image_encoders):
        cond_images = image_processor.preprocess(images, return_tensors='pt').data['pixel_values'].to(image_encoder.device)
        image_embeds = image_encoder(cond_images, output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_image_embeds = image_embeds[0]
        image_embeds = image_embeds.image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.view(bs_embed, seq_len, -1)
        image_embeds_list.append(image_embeds)

    image_embeds = torch.cat(image_embeds_list, dim=-1)
    pooled_image_embeds = pooled_image_embeds.view(bs_embed, -1)
    return image_embeds, pooled_image_embeds


def collate_fn(examples, args, class_labels):
    _image_embeds = torch.stack([example["image_embeds"] for example in examples])
    _pooled_image_embeds = torch.stack([example["pooled_image_embeds"] for example in examples])
    if args.grounding_dino_hidden_states:
        _hidden_states = torch.stack([example["hidden_states"] for example in examples])
    else:
        _hidden_states = None
    _image_latents = torch.stack([example["img_latents"] for example in examples])
    _original_sizes = [example["original_sizes"] for example in examples]
    _crop_top_lefts = [example["crop_top_lefts"] for example in examples]

    labels = []
    label_name_list = []
    label_name_count = []
    label_index = []
    if args.counting_loss or args.class_condition:
        for example in examples:
            image_labels = []
            image_label_index = []
            for instance in example["instances"]:
                label_name = class_labels.int2str(instance["label"])
                image_labels.append(label_name)

            name_count = Counter(image_labels)
            current_index = 0
            for label_name in name_count.keys():
                if ' ' not in label_name:
                    current_index += 1
                    image_label_index.append(current_index)
                    current_index += 1
                else:
                    current_index += 1
                    image_label_index.append([current_index, current_index + 1])
                    current_index += 2

            text_prompt = ". ".join(name_count.keys()) + "." if len(name_count) != 0 else ""
            labels.append(text_prompt)
            label_name_list.append(list(name_count.keys()))
            label_name_count.append(list(name_count.values()))
            label_index.append(image_label_index)

    return {
        "image_embeds": _image_embeds,
        "pooled_image_embeds": _pooled_image_embeds,
        "hidden_states": _hidden_states,
        "image_latents": _image_latents,
        "original_sizes": _original_sizes,
        "crop_top_lefts": _crop_top_lefts,
        "labels": labels,
        "label_name": label_name_list,
        "label_count": label_name_count,
        "label_index": label_index
    }


def collate_fn_train_encoder(examples, args, class_labels):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    images = [example["image"] for example in examples]
    original_sizes = [example["original_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "images": images,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }


def main():
    args = parse_args()
    wandb.require("core")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, InitProcessGroupKwargs(timeout=timedelta(hours=48))]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    train_dataset, image_column, caption_column, class_labels = make_train_dataset(args, accelerator)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_processor2 = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor2")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    image_encoder2 = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if args.counting_loss:
        grounding_dino = GroundingDino("IDEA-Research/grounding-dino-tiny", device=accelerator.device)
    else:
        grounding_dino = None

    if args.grounding_dino_hidden_states:
        gd_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        gd_model = GroundingDinoModel.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        gd_model.requires_grad_(False)
        gd_model.to(accelerator.device)
    else:
        gd_processor = None
        gd_model = None

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    image_encoder2.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder2.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    if args.class_condition:
        from models.class_embed_adapter import ClassEmbedAdapter
        adapter = ClassEmbedAdapter(args, class_labels).to(accelerator.device, dtype=torch.float32)
        unet.class_embedding = adapter

    if args.train_image_encoder:
        image_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        image_encoder.add_adapter(image_lora_config)
        image_encoder2.add_adapter(image_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            unet_class_embedding_to_save = None
            image_encoder_one_lora_layers_to_save = None
            image_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                    if model.class_embedding is not None:
                        unet_class_embedding_to_save = model.class_embedding.state_dict()
                elif unwrap_model(model).config.projection_dim == unwrap_model(image_encoder).config.projection_dim:
                    image_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif unwrap_model(model).config.projection_dim == unwrap_model(image_encoder2).config.projection_dim:
                    image_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLImageVariationPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                image_encoder_lora_layers=image_encoder_one_lora_layers_to_save,
                image_encoder2_lora_layers=image_encoder_two_lora_layers_to_save,
            )
            if unet_class_embedding_to_save is not None:
                torch.save(unet_class_embedding_to_save, os.path.join(output_dir, "unet_class_embedding.pt"))

    def load_model_hook(models, input_dir):
        unet_ = None
        image_encoder_one_ = None
        image_encoder_two_ = None

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif unwrap_model(model).config.projection_dim == unwrap_model(image_encoder).config.projection_dim:
                image_encoder_one_ = model
            elif unwrap_model(model).config.projection_dim == unwrap_model(image_encoder2).config.projection_dim:
                image_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.class_condition:
            unet_class_embedding_state_dict = torch.load(os.path.join(input_dir, "unet_class_embedding.pt"))
            unet.class_embedding.load_state_dict(unet_class_embedding_state_dict)

        if args.train_image_encoder:
            image_encoder_state_dict = {f'{k.replace("image_encoder.", "")}': v for k, v in lora_state_dict.items() if k.startswith("image_encoder.")}
            image_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(image_encoder_state_dict))
            set_peft_model_state_dict(image_encoder_one_, image_encoder_state_dict, adapter_name="default")

            image_encoder2_state_dict = {f'{k.replace("image_encoder2.", "")}': v for k, v in lora_state_dict.items() if k.startswith("image_encoder2.")}
            image_encoder2_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(image_encoder2_state_dict))
            set_peft_model_state_dict(image_encoder_two_, image_encoder2_state_dict, adapter_name="default")

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_image_encoder:
                models.extend([image_encoder_one_, image_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.counting_loss:
            vae.enable_gradient_checkpointing()
        if args.train_image_encoder:
            image_encoder.gradient_checkpointing_enable()
            image_encoder2.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_image_encoder:
            models.extend([image_encoder, image_encoder2])
        cast_training_params(models, dtype=torch.float32)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_image_encoder:
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, image_encoder.parameters()))
            + list(filter(lambda p: p.requires_grad, image_encoder2.parameters()))
        )
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if not args.train_image_encoder:
        # Let's first compute all the latents so that we can free up the Image Encoder from memory.
        image_encoders = [image_encoder, image_encoder2]
        image_processors = [image_processor, image_processor2]
        compute_image_embeddings_fn = functools.partial(
            encode_image if not args.use_content else encode_image_with_content,
            image_encoders=image_encoders,
            image_processors=image_processors,
            vae=vae,
            image_column=image_column,
            args=args,
        )
        if args.grounding_dino_hidden_states:
            compute_hidden_states_fn = functools.partial(
                encode_image_with_grounding_dino,
                processor=gd_processor,
                grounding_dino=gd_model,
                args=args,
                class_labels=class_labels
            )
        else:
            compute_hidden_states_fn = None

        with accelerator.main_process_first():
            from datasets.fingerprint import Hasher

            hash_fingerprint = args.pretrained_model_name_or_path
            if args.resolution == 512:
                hash_fingerprint += "-512"
            if args.use_content:
                hash_fingerprint += "-content"
            if args.use_content and args.content_type != "gt":
                hash_fingerprint += f"-{args.content_type}"
            new_fingerprint_for_image_encoder = Hasher.hash(hash_fingerprint)
            train_dataset_with_image_embeddings = train_dataset.map(
                compute_image_embeddings_fn,
                batched=True,
                batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
                new_fingerprint=new_fingerprint_for_image_encoder,
            )

            precomputed_dataset = train_dataset_with_image_embeddings.with_format(
                type="torch",
                columns=["image_embeds", "pooled_image_embeds", "img_latents"],
                output_all_columns=True
            )

            if args.grounding_dino_hidden_states:
                new_fingerprint_for_grounding_dino = Hasher.hash(args.pretrained_model_name_or_path + "-512-grounding-dino-tiny")
                train_dataset_with_hidden_states = train_dataset.map(
                    compute_hidden_states_fn,
                    batched=True,
                    batch_size=args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
                    new_fingerprint=new_fingerprint_for_grounding_dino,
                )
                if args.dataset == "COCO":
                    remove_columns = ['image', 'image_id', 'file_name', 'height', 'width', 'captions', 'instances']
                else:
                    raise ValueError(f"Unknown dataset {args.dataset}")
                precomputed_dataset = concatenate_datasets(
                    [precomputed_dataset,
                     train_dataset_with_hidden_states.remove_columns(remove_columns)], axis=1
                ).with_format(
                    type="torch",
                    columns=["image_embeds", "pooled_image_embeds", "img_latents", "hidden_states"],
                    output_all_columns=True
                )

                del compute_hidden_states_fn, gd_processor, gd_model

        del compute_image_embeddings_fn, image_encoder, image_encoder2, image_processor, image_processor2
        del image_encoders, image_processors, vae.encoder
        gc.collect()
        torch.cuda.empty_cache()

        image_processor = None
        image_processor2 = None
        image_encoder = None
        image_encoder2 = None

    dataloader_collate_fn = collate_fn if not args.train_image_encoder else collate_fn_train_encoder
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset if not args.train_image_encoder else train_dataset,
        shuffle=True,
        collate_fn=lambda x: dataloader_collate_fn(x, args, class_labels),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_image_encoder:
        unet, image_encoder, image_encoder2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, image_encoder, image_encoder2, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        if args.dataset == "COCO":
            name = "image-variation-sdxl" if args.resolution == 1024 else "image-variation-sdxl-512"
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
        accelerator.init_trackers(
            project_name=name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": args.wandb_run_name
                }
            }
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset if not args.train_image_encoder else train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_image_encoder:
            image_encoder.train()
            image_encoder2.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Load text and image embeds
                if not args.train_image_encoder:
                    image_embeds = batch["image_embeds"].to(accelerator.device, dtype=weight_dtype)
                    if not args.grounding_dino_hidden_states:
                        pooled_image_embeds = batch["pooled_image_embeds"].to(accelerator.device, dtype=weight_dtype)
                    else:
                        pooled_image_embeds = batch["hidden_states"].to(accelerator.device, dtype=weight_dtype)
                    image_latents = batch["image_latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    image_latents = vae.encode(pixel_values).latent_dist.sample()
                    image_latents = image_latents * vae.config.scaling_factor
                    del pixel_values

                    image = batch["images"]
                    image_processors = [image_processor, image_processor2]
                    image_encoders = [image_encoder, image_encoder2]
                    image_embeds, pooled_image_embeds = encode_image_without_batch(image_processors, image_encoders, image)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(image_latents)
                bsz = image_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=image_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                image_latents = noise_scheduler.add_noise(image_latents, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = [args.resolution, args.resolution]
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": time_ids, "text_embeds": pooled_image_embeds}
                model_pred = unet(
                    image_latents,
                    timesteps,
                    encoder_hidden_states=image_embeds,
                    class_labels=batch["label_name"] if args.class_condition else None,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(image_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.counting_loss and global_step >= args.counting_loss_steps:
                    if args.denoise_type == "minus":
                        final_output = image_latents - model_pred
                    elif args.denoise_type == "step":
                        if bsz == 1:
                            final_output = noise_scheduler.step(model_pred, timesteps, image_latents, return_dict=False)[0]
                        else:
                            final_output = torch.cat(
                                [noise_scheduler.step(model_pred[i:i+1, ...], timesteps[i], image_latents[i:i+1, ...], return_dict=False)[0] for i in range(bsz)],
                                dim=0
                            )
                    else:
                        raise ValueError

                    final_output = final_output.to(accelerator.device, dtype=weight_dtype)
                    final_output = vae.decode(final_output / vae.config.scaling_factor, return_dict=False)[0]
                    final_output = torch.stack([(final_output[i] / 2 + 0.5).clamp(0, 1) for i in range(final_output.shape[0])])

                    counting_loss = grounding_dino(
                        final_output,
                        batch["labels"],
                        batch["label_count"],
                        batch["label_index"],
                        args.counting_loss_threshold,
                        args.counting_loss_scale
                    ) / bsz

                    loss += counting_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Back propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            image_encoder,
                            image_encoder2,
                            class_labels=class_labels
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_image_encoder:
            image_encoder = unwrap_model(image_encoder)
            image_encoder2 = unwrap_model(image_encoder2)
            image_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(image_encoder))
            image_encoder2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(image_encoder2))
        else:
            image_encoder_lora_layers = None
            image_encoder2_lora_layers = None

        StableDiffusionXLImageVariationPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            image_encoder_lora_layers=image_encoder_lora_layers,
            image_encoder2_lora_layers=image_encoder2_lora_layers,
        )
        if args.class_condition:
            assert unet.class_embedding is not None
            torch.save(unet.class_embedding.state_dict(), os.path.join(args.output_dir, "unet_class_embedding.pt"))

        # Run a final round of validation.
        log_validation(
            unet=None,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=global_step,
            is_final_validation=True,
            image_encoder=None,
            image_encoder2=None,
            class_labels=class_labels
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
