#!/usr/bin/env bash

if [[ $# -eq 0 ]] ; then
    echo 'Please provide an argument for k (current fold)'
    exit 0
fi

source deactivate
source activate thesis_cluster2
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --augment --hist"

declare -a StringArray=("noise", "intensity", "rand_resize", "rotate", "translate")
for val in ${StringArray[@]}; do
  bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 "python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --${val}"


bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --noise --intensity --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --intensity --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --rand_resize --rotate --translate""

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --noise --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --noise --intensity --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --hist --rotate --translate"