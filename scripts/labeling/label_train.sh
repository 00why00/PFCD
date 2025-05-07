#!/usr/bin bash

export CUDA_VISIBLE_DEVICES=1
## shellcheck disable=SC2045
for name in $(ls data/COCO-YOLO/images/train)
do
    echo "$name"
    python utils/create_coco_dataset.py --image_directory data/COCO-YOLO/images/train/"$name" --export_path data/Generation-train
done