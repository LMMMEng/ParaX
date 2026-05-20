#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/../.." && pwd)

NUM_GPUS=${NUM_GPUS:-4}
PORT=${PORT:-29500}
PRETRAINED=${PRETRAINED:-$REPO_ROOT/pretrained_models/mae_pretrain_vit_b.pth}
CLS_DATA_ROOT=${CLS_DATA_ROOT:-$REPO_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/work_dirs/imagenet-r/parax}

python -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$PORT" \
  "$PROJECT_DIR/main_image_parax.py" \
  --batch_size 256 \
  --cls_token \
  --finetune "$PRETRAINED" \
  --dist_eval \
  --data_path "$CLS_DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --drop_path 0.1 \
  --parax_enable_conv \
  --parax_kernel_sizes "(3, 5, 7)" \
  --parax_router_hidden 16 \
  --blr 0.1 \
  --weight_decay 1e-4 \
  --dataset imagenet-r \
  "$@"