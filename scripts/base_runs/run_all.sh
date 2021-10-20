#!/usr/bin/env bash

# This bash script expects one argument, k=current fold.
# It then runs all the baseline experiments for this fold

if [[ $# -eq 0 ]] ; then
    echo 'Please provide an argument for k (current fold)'
    exit 0
fi
echo "Running script for fold no: $1"
./scripts/base_runs/no_aug.sh $1
./scripts/base_runs/hist_only.sh $1
./scripts/base_runs/aug_no_hist_all_combos.sh $1
./scripts/base_runs/aug_hist_all_combos.sh $1
