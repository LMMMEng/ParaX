#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config> <checkpoint> [gpus] [extra args...]"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=$1
CHECKPOINT=$2
GPUS=1
PORT=${PORT:-29500}

if [[ $# -ge 3 && "$3" =~ ^[0-9]+$ ]]; then
  GPUS=$3
  shift 3
else
  shift 2
fi

python -m torch.distributed.run \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  "$PROJECT_DIR/test.py" \
  "$CONFIG" \
  "$CHECKPOINT" \
  --launcher pytorch \
  "$@"