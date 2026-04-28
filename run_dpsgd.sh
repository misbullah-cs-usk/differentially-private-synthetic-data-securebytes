#!/bin/bash

python3 dpsgd_mlp_adult.py \
  --adult_path /home/alim/PhD-Experiments/Alim/DataSecurity/HW1/adult_dataset/adult.data \
  --epochs 20 \
  --batch_size 250 \
  --num_microbatches 250 \
  --learning_rate 0.05 \
  --l2_norm_clip 1.0 \
  --noise_multipliers 0.5 1.0 1.5 2.0 3.0 \
  --output_dir dpsgd_mlp_adult_outputs
