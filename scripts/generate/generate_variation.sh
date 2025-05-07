#!/usr/bin bash

export CUDA_VISIBLE_DEVICES=3,5,6,7

accelerate launch --num_processes 4 utils/test_utils.py --train_name content-counting_loss-th0.1-step1000-scale0.5 --use_content