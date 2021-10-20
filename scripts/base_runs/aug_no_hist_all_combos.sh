#!/usr/bin/env bash

source deactivate
source activate thesis_cluster2
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --augment --k $1'

declare -a StringArray=("noise", "intensity", "rand_resize", "rotate", "translate")
for val in ${StringArray[@]}; do
  bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
  'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --${val}'


bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --intensity --rand_resize --rotate --translate'

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --intensity --rand_resize --rotate --translate'

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --rand_resize --rotate --translate'

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --rand_resize --rotate --translate'

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --intensity --rotate --translate'

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
'python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --rotate --translate'
