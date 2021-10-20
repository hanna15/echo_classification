#!/usr/bin/env bash

# This bash script expects one argument, k=current fold.
# It then runs all the baseline experiments for this fold

echo "Running script for fold no: $1"
./base_runs/no_aug.sh $1
./base_runs/hist_only.sh $1
./base_runs/aug_no_hist_all_combos.sh $1
./base_runs/aug_hist_all_combos.sh $1