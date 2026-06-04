#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=2
source /data/m25_aditya_psingh/miniconda3/bin/activate seq2json_lora_env
python - <<'PY'
import torch
print(f'CUDA_VISIBLE_DEVICES={__import__("os").environ.get("CUDA_VISIBLE_DEVICES")}')
print(f'visible_gpu_count={torch.cuda.device_count()}')
if torch.cuda.device_count() != 1:
    raise SystemExit(f'Expected exactly 1 visible GPU, found {torch.cuda.device_count()}')
PY
python train_and_eval_custom_models.py \
  --output-dir artifacts_full_dataset
