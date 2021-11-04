#!/usr/bin/env bash

if [[ $# -le 1 ]] ; then
    echo 'Please provide an argument for k (current fold) and dropout value'
    exit 0
fi

echo "starting jobs for training on fold $1"
source deactivate
source activate thesis_cluster2

echo "Starting a run for fold $1, with dropout value $2"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --augment --hist --class_balance_per_epoch --model res_simple --dropout $2 --log_freq 1 --max_epochs 300 --optimizer adamw"

