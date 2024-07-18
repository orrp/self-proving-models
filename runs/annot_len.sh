#!/bin/bash
cd ../spm || exit 1

for arg in "$@"; do
  if [ "$arg" == "--dry_run" ]; then
    DRY_RUN=1
  fi
done

# datasets are of size 10240000. 10 epochs are 100k iterations for batch size 1024
# eval_interval=10000 means 10 evaluations total (1 per epoch).
SEEDS="0 1 2"
EPOCHS="10"
BETA1="0.733"
LR="0.0007"
BATCH_SIZE=1024
DECAY_LR=10
GRAD_CLIP=2
N_EMBD=256
N_HEAD=8
N_LAYER=8
WANDB_PROJ="self-proving-models"
EVAL_INTERVAL="10000"
LOG_INTERVAL="500"

DATA="1e4_m1e7_b210"

if [ "$DRY_RUN" ] ; then
  echo ">>>> DRY RUN"
  SEEDS="999"
  EPOCHS="4"
  EVAL_INTERVAL="2"
  LOG_INTERVAL="1"
  N_HEAD=1
  N_LAYER=1
  N_EMBD=32
  WANDB_PROJ="self-proving-models-test"
  DATA="1e4_m1e4_b210"
fi

BASE_CMD="python train.py --wandb --wandb_proj $WANDB_PROJ --device=cuda --dropout=0 --eval_batch_size=512 --eval_interval=$EVAL_INTERVAL --log_interval=$LOG_INTERVAL --warmup_iters=0"
HYPERS="--epochs=$EPOCHS --beta1=$BETA1 --learning_rate=$LR --batch_size=$BATCH_SIZE --decay_lr=$DECAY_LR --grad_clip=$GRAD_CLIP --n_embd=$N_EMBD --n_head=$N_HEAD --n_layer=$N_LAYER"
BASE_CMD="$BASE_CMD $HYPERS"

for seed in $SEEDS; do
  cmd="$BASE_CMD --seed=$seed"
  # if seed is 0, save the model at the end of training
  if [ "$seed" -eq 0 ]; then
    cmd="$cmd --save_iters -1"
  fi
  $cmd "--data=Baseline_$DATA" || exit # Table 1
  $cmd "--data=TL_$DATA" || exit # Table 1, Figure 2, 4
  # Annotated Transcript Learning (varying annotation cutoff T=3,4,5,6,7)
  $cmd "--data=ATL7_$DATA" || exit  # Table 1, Figure 2, 4. This one needs the most CUDA mem.
  $cmd "--data=ATL3_$DATA" || exit  # Figure 2, 4
  $cmd "--data=ATL4_$DATA" || exit  # Figure 4
  $cmd "--data=ATL5_$DATA" || exit  # Figure 2, 4
  $cmd "--data=ATL6_$DATA" || exit  # Figure 4
done
