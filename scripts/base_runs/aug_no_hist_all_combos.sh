#!/usr/bin/env bash

if [[ $# -eq 0 ]] ; then
    echo 'Please provide an argument for k (current fold)'
    exit 0
fi

echo "training on fold $k"
source deactivate
source activate thesis_cluster2
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 "python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --augment --k $1"

echo "finished starting training on all augments"

declare -a StringArray=("noise" "intensity" "rand_resize" "rotate" "translate")
for val in ${StringArray[@]}; do
  echo "starting training on only specific augment: ${val}"
  bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 "python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --${val}"
done
exit 0
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --intensity --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --intensity --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --rand_resize --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --noise --intensity --rotate --translate"

bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 04:00 -n 6 \
	"python scripts/train_simple.py --cache_dir ~/.heart_echo --label_type 2class_drop_ambiguous --k $1 --rotate --translate"

