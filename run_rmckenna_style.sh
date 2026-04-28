#!/bin/bash

python3 rmckenna_dp_synth_adult.py \
  --adult_path /home/alim/PhD-Experiments/Alim/DataSecurity/HW1/adult_dataset/adult.data \
  --epsilon 0.1 0.5 1.0 2.0 5.0 \
  --delta 1e-5 \
  --top_pairs 20 \
  --top_triples 8 \
  --output_dir rmckenna_dp_synth_adult_outputs
