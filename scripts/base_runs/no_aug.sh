#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 "python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1"
