#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 TARGET_FLOPS{8.7,10.0,10.4,11.5}"
    exit 1
fi

# batch-size of 128 results in 22GB memory usage on RTX 3090
python \
    -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29503 \
    main.py \
        --arch-lr 0.01 --arch-min-lr 0.001 \
       	--autoresume \
       	--dist-eval \
        --data-set MSRVTT \
        --data-path /mnt/ssd2/dataset \
        --output_dir /mnt/ssd3/diffrate/msrvtt/$1 \
        --batch-size 128 \
        --model vit_base_patch16_clip_224.openai \
        --alpha 1 \
        --target_flops $1 # {8.7,10.0,10.4,11.5}
