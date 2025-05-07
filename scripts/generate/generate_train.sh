#!/usr/bin bash

export CUDA_VISIBLE_DEVICES=1,2

accelerate launch --num_processes 2 utils/test_utils.py --split train --train_name content-counting_loss-th0.1-step1000-scale0.5 --use_content