#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 args..."
    exit 1
fi

TRANSFORMERS_CACHE=microsoft \
/opt/conda/bin/python3.8 \
    mc.py \
    --test --eval \
    --combine_datasets how2qa \
    --combine_datasets_val how2qa \
    --how2qa_val_csv_path /mnt/hdd2/how2qa/frozenbilm/public_val_sanitized.csv \
    --how2qa_subtitles_path /mnt/hdd2/how2qa/frozenbilm/subtitles.pkl \
    --save_dir=zsHow2QA \
    --ds_factor_ff=8 \
    --ds_factor_attn=8 \
    --suffix="." \
    --max_tokens=512 \
    --load=/mnt/ssd1/data/how2qa/frozenbilm_how2qa.pth \
     --print_freq 5 \
     "$@"
