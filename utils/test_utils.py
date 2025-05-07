import argparse
import os
import random
import sys
from collections import Counter

import datasets
import torch
from accelerate import PartialState
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed, GroundingDinoProcessor, GroundingDinoModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipelines.pipeline_stable_xl_image_variation import StableDiffusionXLImageVariationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL Image Variation Training")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="inference/saved_pipeline/stable-xl-image-variation",
        help="Path to trained model.",
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
        help="The name of training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='COCO',
    )
    parser.add_argument(
        "--split",
        type=str,
        default='validation',
        help="The split of dataset to use."
    )
    parser.add_argument(
        "--use_content",
        action="store_true",
        help="Whether to use content dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/{}-YOLO/images/{}",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--grounding_dino_hidden_states",
        action="store_true",
        help="Whether to use hidden states from DINO."
    )
    parser.add_argument(
        "--class_condition",
        action="store_true",
        help="Whether to use class condition."
    )
    args = parser.parse_args()
    set_seed(args.seed)
    if args.split == "validation":
        args.output_dir = args.output_dir.format(args.dataset.upper(), args.train_name)
    else:
        assert args.split == "train"
        args.output_dir = args.output_dir.format(args.dataset.upper(), "train/" + args.train_name)
    os.makedirs(args.output_dir, exist_ok=True)

    return args

def make_val_dataset(args):
    if args.dataset == "COCO":
        file_name = 'coco.py'
    elif args.dataset == "VOC":
        file_name = 'voc.py'
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if args.split == "validation":
        dataset = load_dataset(f"dataset/{file_name}", split="validation", trust_remote_code=True)
    else:
        assert args.split == "train"
        dataset = load_dataset(f"dataset/{file_name}", split="train", trust_remote_code=True)
        idx = [i for i in range(len(dataset))]
        random.shuffle(idx)
        dataset = dataset.select(idx[:10000])
        dataset = dataset.flatten_indices()
    image_column = "image"
    caption_column = "captions"
    class_labels = dataset.features['instances'][0]['label']
    assert isinstance(dataset, datasets.Dataset), "Dataset should be of type `datasets.Dataset`"
    return dataset, image_column, caption_column, class_labels

def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load dataset
    val_dataset, image_column, caption_column, class_labels = make_val_dataset(args)

    # Load model
    distributed_state = PartialState()
    if args.grounding_dino_hidden_states:
        gd_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        gd_model = GroundingDinoModel.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
    else:
        gd_processor = None
        gd_model = None

    pipeline = StableDiffusionXLImageVariationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        gd_processor=gd_processor,
        gd_model=gd_model,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    lora_wight_dir = f"output/image-variation-sdxl-{args.train_name}"
    pipeline.load_lora_weights(lora_wight_dir)
    if args.class_condition:
        from models.class_embed_adapter import ClassEmbedAdapter
        adapter = ClassEmbedAdapter(args, class_labels)
        pipeline.set_class_embedding_adapter(adapter)
        pipeline.load_class_embedding_adapter(os.path.join(lora_wight_dir, "unet_class_embedding.pt"))
    pipeline = pipeline.to(distributed_state.device, dtype=torch.float16)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(distributed_state.device).manual_seed(args.seed)
    with distributed_state.split_between_processes(val_dataset) as split_dataset:
        for batch in tqdm(split_dataset):
            image = batch[image_column].convert("RGB")
            if args.class_condition or args.grounding_dino_hidden_states:
                image_labels = []
                for instance in batch["instances"]:
                    label_name = class_labels.int2str(instance["label"])
                    image_labels.append(label_name)
                name_count = Counter(image_labels)
                label_name_list = [list(name_count.keys())]
                text_prompt = ". ".join(name_count.keys()) + "." if len(name_count) != 0 else ""
            else:
                label_name_list = None
                text_prompt = None

            out = pipeline(
                image=image,
                height=512,
                width=512,
                generator=generator,
                use_content=args.use_content,
                image_info=batch,
                class_labels=label_name_list,
                class_label = text_prompt
            ).images[0]
            out.save(os.path.join(args.output_dir, batch["file_name"]))


if __name__ == '__main__':
    main()
