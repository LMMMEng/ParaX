#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config> [gpus] [extra args...]"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=$1
GPUS=4
PORT=${PORT:-29500}

if [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
  GPUS=$2
  shift 2
else
  shift 1
fi

python -m torch.distributed.run \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  "$PROJECT_DIR/train.py" \
  "$CONFIG" \
  --launcher pytorch \
  "$@"