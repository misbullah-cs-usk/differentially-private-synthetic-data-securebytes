#!/bin/bash

python3 dpsyn_style_adult.py \
  --adult_path /home/alim/PhD-Experiments/Alim/DataSecurity/HW1/adult_dataset/adult.data \
  --epsilons 0.1 0.5 1.0 2.0 5.0 \
  --delta 1e-5 \
  --top_pairs 25 \
  --top_triples 10 \
  --random_pairs 15 \
  --random_triples 8 \
  --consistency_rounds 8 \
  --repair_passes 6 \
  --output_dir dpsyn_style_adult_outputs
