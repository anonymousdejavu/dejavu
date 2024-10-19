#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 TARGET_FLOPS{31.0,34.7,38.5,42.3,46.1}"
    exit 1
fi

# batch-size of 30 results in 22GB memory usage on RTX 3090
python \
    -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29515 \
    main.py \
        --arch-lr 0.01 --arch-min-lr 0.001 \
        --autoresume \
        --dist-eval \
        --data-set HOW2QA \
        --data-path /mnt/ssd2/dataset \
        --output_dir /mnt/ssd3/diffrate/how2qa/$1 \
        --batch-size 30 \
        --model vit_large_patch14_clip_224.openai \
        --alpha 1 \
        --target_flops $1 \
        --epoch 20
