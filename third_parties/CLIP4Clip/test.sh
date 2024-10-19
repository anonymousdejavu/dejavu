#!/bin/bash

if [[ $# -lt 1 ]] ; then
    echo 'Usage: $0 args...'
    exit 1
fi

export DATA_PATH=/mnt/hdd2/MSRVTT/msrvtt_data
python -m torch.distributed.launch \
	--master_port 0 \
	main_task_retrieval.py \
    --do_eval \
    --num_thread_reader=0 \
    --n_display=50 \
    --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/MSRVTT_data.json \
    --features_path ${DATA_PATH}/MSRVTT_Videos \
    --output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
    --max_frames 12 \
    --batch_size_val 16 \
    --datatype msrvtt \
    --expand_msrvtt_sentences  \
    --feature_framerate 1 \
    --freeze_layer_num 0 \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header meanP \
    --pretrained_clip_name ViT-B/16 \
    "$@"
