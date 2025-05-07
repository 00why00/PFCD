#!/usr/bin bash

export MODEL_NAME="inference/saved_pipeline/stable-xl-image-variation"
export CUDA_VISIBLE_DEVICES=1,2

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 2 --main_process_port 19919 train_image_variation_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME --dataset="COCO" --dataset_type="coco" \
  --resolution=512 --center_crop --random_flip --mixed_precision="fp16" \
  --train_batch_size=1 --num_train_epochs=5 --denoise_type=step --use_content \
  --counting_loss --counting_loss_threshold=0.1 --counting_loss_scale=0.5 --counting_loss_steps=1000 \
  --gradient_accumulation_steps=16 --use_8bit_adam --rank=128 \
  --learning_rate=1e-4 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 --validation_steps=1000 \
  --wandb_run_name=content-counting_loss-th0.1-step1000-scale0.5 --output_dir="output/image-variation-sdxl-content-counting_loss-th0.1-step1000-scale0.5"



