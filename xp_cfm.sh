#!/bin/bash

# Define the ratios and seeds
ratios=(1 4 10 40 100)
seeds=(1 2 3 4 5)

for seed in "${seeds[@]}"; do
  echo "seed ${seed}"
  for ratio in "${ratios[@]}"; do
    python xp_swggcfm.py "$ratio" "$seed"
  done
#   python xp_otcfm.py "$seed"
#   python xp_cfm.py "$seed"
done
