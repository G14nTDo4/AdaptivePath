#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash train_phase2.sh DATA_ROOT PHASE1_CHECKPOINT OUTPUT_DIR [CUDA_DEVICE]"
    exit 2
fi
DATA_ROOT=$1
PHASE1_CHECKPOINT=$2
OUTPUT_DIR=$3
CUDA_DEVICE=${4:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" torchrun \
    --nproc_per_node=1 --master_port=29616 main_ppo.py \
    --data_path "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --finetune "$PHASE1_CHECKPOINT" \
    --lr 5e-5 --batch_size 32 --seq_l 4 --clip_grad 0.5 \
    --input_size 512 --glance_input_size 112 --focus_patch_size 112 \
    --focus_scale_min 0.25 --focus_scale_max 1.0 \
    --spatial_hflip 0.5 --spatial_vflip 0.5 \
    --spatial_affine_prob 0.8 \
    --spatial_degrees 15.0 \
    --spatial_translate 0.12 \
    --spatial_scale_min 0.90 --spatial_scale_max 1.10 \
    --focus_net_reg_size 160 --loss_reg_focus_net_weight 1.0 \
    --gamma 0.5 --ppo_lam 0.84 --ppo_update_steps 5 --update_policy_freq 10 \
    --focus_consistency_weight 0.05 \
    --focus_consistency_every_n_steps 10 \
    --focus_consistency_max_batch 8 \
    --focus_consistency_heatmap_size 32 \
    --focus_consistency_sigma 0.10 \
    --focus_consistency_geo_weight 1.0 \
    --focus_consistency_step_geo_weight 0.5 \
    --focus_consistency_app_weight 0.5 \
    --focus_consistency_scale_weight 0.2 \
    --focus_consistency_hflip 0.5 --focus_consistency_vflip 0.5 --focus_consistency_rotate90 1.0 \
    --focus_consistency_app_brightness 0.15 \
    --focus_consistency_app_contrast 0.15 \
    --focus_consistency_app_saturation 0.10 \
    --num_workers 8 \
    --mixup 0.2 \
    --weight_decay 0.02 \
    --epochs 100 \
    --warmup_epochs 5 \
