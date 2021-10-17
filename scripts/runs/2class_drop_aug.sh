source deactivate
source activate thesis_cluster2
pip install -e .
bsub -R "rusage[mem=4096, ngpus_excl_p=1]" -W 08:00 -n 6 "python scripts/train_simple.py --cache_dir ~/.heart_echo --scaling_factor 0.25 --label_type 2class_drop_ambiguous --batch_size 128 --max_epochs 1000 --class_balance_per_epoch --augment"