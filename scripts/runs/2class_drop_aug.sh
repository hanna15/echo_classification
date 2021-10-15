source deactivate
source activate thesis_cluster
python scripts/train_simple.py --cache_dir ~/.heart_echo --scaling_factor 0.25 --label_type 2class_drop_ambiguous --batch_size 30 --max_epochs 1000 --class_balance_per_epoch --augment