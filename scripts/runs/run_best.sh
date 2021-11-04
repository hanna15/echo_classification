#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2

  for fold in {0..9}; do
    echo "Starting run for fold ${fold}"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --res_dir results_best --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --augment --aug_type 3 --optimizer adam --pretrained"
    #bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #  "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --res_dir results_best --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --optimizer adamw --pretrained"
    bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
      "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --res_dir results_best --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --augment --aug_type 4 --optimizer adam --pretrained"
    #bsub -R "rusage[mem=2048, ngpus_excl_p=1]" -W 04:00 -n 6 \
    #  "python scripts/train_simple.py --max_epochs 300 --max_p 95 --batch_size 64 --wd 0.0001 --res_dir results_best --eval_metrics video-b-accuracy/valid --cache_dir ~/.heart_echo --k 10 --fold ${fold} --hist --class_balance_per_epoch --augment --aug_type 2 --optimizer adamw --pretrained"
  done
done