#!/bin/bash
DIRS_TO_BACKUP="logs data models"

echo "Starting dry run..."
time_start=$(date +%s)
backup_suffix=$(date +%Y%m%d%H%M%S)
cd ..
echo "Backing up $DIRS_TO_BACKUP..."
mkdir -p backup || exit
for dir in $DIRS_TO_BACKUP; do
  # if dir not found, create it
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir" || exit
  fi
  mv "$dir" backup/"${dir}"_"$backup_suffix" || exit
done
cd spm || exit
# generate data
echo "Generating data..."
python data/generate_data.py --n_train 10240 || exit
# run T_ablation.sh with --dry_run
echo "Running T_ablation.sh with --dry_run..."
cd ../runs || exit
bash annot_len.sh --dry_run || exit
# run bases experiment
echo "Running bases experiment..."
cd ../spm || exit
python train_diff_bases.py --seed 0 --num_unique_primes 2 --dry_run || exit
# restore logs
cd ..
echo "Restoring $DIRS_TO_BACKUP..."
for dir in $DIRS_TO_BACKUP; do
  rm -rf "$dir"
  mv backup/"${dir}"_"$backup_suffix" "$dir" || exit
done

time_end=$(date +%s)
echo "Dry run complete"
echo "Time elapsed: $((time_end - time_start)) seconds"
