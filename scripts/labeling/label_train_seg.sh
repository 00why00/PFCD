#!/usr/bin bash

export CUDA_VISIBLE_DEVICES=3
# shellcheck disable=SC2045
for name in $(ls data/COCO-YOLO/images/train)
do
    echo "$name"
    python utils/create_coco_dataset_with_seg.py --image_directory data/COCO-YOLO/images/train/"$name" --export_path data/Generation-train-seg
done