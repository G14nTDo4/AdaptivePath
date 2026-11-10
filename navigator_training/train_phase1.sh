#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash train_phase1.sh DATA_ROOT OUTPUT_DIR [CUDA_DEVICE]"
    exit 2
fi
DATA_ROOT=$1
OUTPUT_DIR=$2
CUDA_DEVICE=${3:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" torchrun \
    --nproc_per_node=1 --master_port=29615 main_ppo.py \
    --data_path "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --lr 1e-4 --batch_size 32 --seq_l 4 --clip_grad 0.5 \
    --input_size 512 --glance_input_size 112 --focus_patch_size 112 \
    --focus_scale_min 0.25 --focus_scale_max 1.0 \
    --spatial_hflip 0.5 --spatial_vflip 0.5 \
    --spatial_affine_prob 0.8 \
    --spatial_degrees 15.0 \
    --spatial_translate 0.12 \
    --spatial_scale_min 0.90 --spatial_scale_max 1.10 \
    --focus_net_reg_size 160 --loss_reg_focus_net_weight 1.0 \
    --gamma 0.5 --ppo_lam 0.84 --ppo_update_steps 5 --update_policy_freq 10 \
    --focus_consistency_weight 0.0 \
    --num_workers 8 \
    --mixup 0.2 \
    --weight_decay 0.02 \
    --epochs 300 \
    --warmup_epochs 10 \
