# Déjà VU: Accelerating Video Understanding by Leveraging Token Similarities in Vision Transformer

## Table of Contents
- [Supported End Tasks](#supported-end-tasks)
- [Preparing Datasets](#preparing-datasets)
- [DiffRate Baseline](#diffrate-baseline)
- [ReuseViT](#reusevit)
## Supported End Tasks
1. Video Retrieval: CLIP4Clip - MSRVTT
   - Base model: ViT-B/16
   - Trancode: 224x224 / 2 FPS
   - Available splits
      - `train`: train split
      - `test`: test split
2. VideoQA: FrozenBiLM - How2QA
   - Base model: ViT-L/14
   - Trancode: 256x256 / 2 FPS
   - Available splits
      - `train`: train split, but it's too large, we are using `test` split for training instead.
      - `frozenbilm`: split provided by FrozenBiLM repository, subset of validation split.
      - `test`: test split, confirmed no overlapping with `frozenbilm` split.

## Preparing Datasets
All end tasks need the same following steps.
1-1. Downloading transcoded videos from wandb
If this is your first ime, you should follow 1-2 upto 2-1
```
python -m dejavu.preprocess.wandb.download_videos --dataset msrvtt --split train --dry-run
python -m dejavu.preprocess.wandb.download_videos --dataset msrvtt --split test

python -m dejavu.preprocess.wandb.download_videos --dataset how2qa --split test 
python -m dejavu.preprocess.wandb.download_videos --dataset how2qa --split frozenbilm 
```

1-2. Downloading videos from YouTube
   - How2QA: Donwload videos from Youtube using the following command
      ```
       python -m dejavu.preprocess.download --help
       python -m dejavu.preprocess.download <dataset> <split>
      ```
   - MSRVTT: The raw videos can be found in [sharing](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt) from Frozen️ in Time, using the following command.
      ```
      wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
      ```

2. Transcode videos
    ```
    python -m dejavu.preprocess.transcode --help
    python -m dejavu.preprocess.transcode how2qa test 1
    python -m dejavu.preprocess.transcode how2qa frozenbilm 1
    python -m dejavu.preprocess.transcode msrvtt train 2
    python -m dejavu.preprocess.transcode msrvtt test 2

2-1. (Optional) Uploading transcoded videos to wandb
```
python -m dejavu.preprocess.wandb.upload_videos --dataset msrvtt --split train
python -m dejavu.preprocess.wandb.upload_videos --dataset msrvtt --split test

python -m dejavu.preprocess.wandb.upload_videos --dataset how2qa --split test
python -m dejavu.preprocess.wandb.upload_videos --dataset how2qa --split test
python -m dejavu.preprocess.wandb.upload_videos --dataset how2qa --split frozenbilm
```

3. Extract CLIP features
```
python -m dejavu.preprocess.extract how2qa frozenbilm 2 openai/clip-vit-large-patch14 --target-features 'i,o' --num-workers 32 
python -m dejavu.preprocess.extract msrvtt test 2 openai/clip-vit-base-patch16 --target-features 'i,o,h' --dry-run
```

   - Extracts 4 types of features, `pixel`, `input`, `output`, `hidden`
     - `pixel`(`p`): Center cropped & resized RGB frames.
     - `input`(`i`): Center cropped & resized & normalized RGB frames.
     - `output`(`o`): Feature from CLIP, including output projection.
     - `hidden states`(`h`): Hidden states from the last encoders, before output projection.
   - Note we are using HuggingFace model unlike FrozenBiLM, but the results are confirmed to be identical on `jupyter/frozenbilm.ipynb`.
   - Involves decoding videos and sampling frames from them.
   - Most of the time comes from video decoding, thus we use cluster server with no GPU.
    ```
    cd /workspace
    python -m dejavu.preprocess.extract --help
    python -m dejavu.preprocess.extract <dataset> <split> <fps> <base_model_name> --num-workers <num_workers> --num-gpus <num_gpus>
    python -m dejavu.preprocess.extract how2qa test 2 openai/clip-vit-large-patch14 --num-workers 6 --num-gpus 0
    python -m dejavu.preprocess.extract msrvtt train 2 openai/clip-vit-base-patch16 --num-workers 6 --num-gpus 0
    ```

## Measure end task accuracy
1. Video Retrieval: CLIP4Clip - MSRVTT
  - Run inside CLIP4Clip container
  ```
  cd /workspace/third_parties/CLIP4Clip
  ./test.sh original 29500 0
  ```
  Result: T2V R@1 40.7, R@5 68.8 R@10 79.8, V2T R@1 39.5 R@5 68.1 R@10 77.7

2. VideoQA: FrozenBiLM - How2QA
  - Run inside FrozenBiLM container
  ```
  cd /workspace/third_parties/FrozenBiLM
  ./test_how2qa.sh original
  ```
  Result: 85.27

## DiffRate Baseline
### Train DiffRate
Inside DiffRate container, run
```
cd /workspace/third_parties/DiffRate/
./train_msrvtt.sh 8.7 # {8.7, 10.0, 10.4, 11.5}
./train_how2qa.sh 31.0 # {31.0,34.7,38.5,42.3,46.1}
```

### Extract using DiffRate
```
cd /workspace
python -m dejavu.preprocess.extract_dataset --help

python -m dejavu.preprocess.extract_dataset --mode diffrate --dataset msrvtt --diffrate-target-flops 8.7 --fps 2
python -m dejavu.preprocess.extract_dataset --mode diffrate --dataset msrvtt --diffrate-target-flops 10.0 --fps 2
python -m dejavu.preprocess.extract_dataset --mode diffrate --dataset msrvtt --diffrate-target-flops 10.4 --fps 2
python -m dejavu.preprocess.extract_dataset --mode diffrate --dataset msrvtt --diffrate-target-flops 11.5 --fps 2
   Saved in /mnt/ssd2/dataset/msrvtt/feature/openai_clip-vit-base-patch16/fps2/test/diffrate/original-8.7
```
### Test end task with DiffRate featuers
1. Video Retrieval: CLIP4Clip - MSRVTT
  - Run inside CLIP4Clip container
  ```
  cd /workspace/third_parties/CLIP4Clip
  ./test.sh diffrate-original-8.7 29500 0
  ./test.sh diffrate-original-10.0 29501 1
  ./test.sh diffrate-original-10.4 29502 2
  ./test.sh diffrate-original-11.5 29503 3

  ./test.sh diffrate/msrvtt/8.7 29500 0
  ./test.sh diffrate/msrvtt/10.0 29501 1
  ./test.sh diffrate/msrvtt/10.4 29502 2
  ./test.sh diffrate/msrvtt/11.5 29503 3
  ```
  Result
   - original-8.7:  T2V R@1 35.2 R@5 61.4 R@10 72.0, V2T R@1 35.1 R@5 62.9 R@10 71.9
   - original-10.0: T2V R@1 38.1 R@5 64.5 R@10 74.5, V2T R@1 37.9 R@5 67.3 R@10 77.0
   - original-10.4: T2V R@1 39.7 R@5 64.7 R@10 75.3, V2T R@1 39.0 R@5 67.1 R@10 77.6
   - original-11.5: T2V R@1 40.5 R@5 64.6 R@10 75.1, V2T R@1 40.5 R@5 68.7 R@10 79.0

   - finetune-8.7:  T2V R@1 36.7 R@5 66.3 R@10 75.9, V2T R@1 35.8 R@5 64.5 R@10 75.7
   - finetune-10.0: T2V R@1 39.0 R@5 68.0 R@10 77.3, V2T R@1 36.6 R@5 66.8 R@10 77.6
   - finetune-10.4: T2V R@1 39.5 R@5 68.0 R@10 77.8, V2T R@1 37.5 R@5 67.3 R@10 77.7
   - finetune-11.5: T2V R@1 41.1 R@5 68.2 R@10 77.4, V2T R@1 39.0 R@5 68.4 R@10 77.9

   - original:      T2V R@1 40.7 R@5 68.8 R@10 79.8, V2T R@1 39.5 R@5 68.1 R@10 77.7

2. VideoQA: FrozenBiLM - How2QA
  - Run inside FrozenBiLM container
  ```
  cd /workspace/third_parties/FrozenBiLM
  CUDA_VISIBLE_DEVICES=0 ./test_how2qa.sh diffrate-original-31.0
  CUDA_VISIBLE_DEVICES=1 ./test_how2qa.sh diffrate-original-34.7
  CUDA_VISIBLE_DEVICES=2 ./test_how2qa.sh diffrate-original-38.5
  CUDA_VISIBLE_DEVICES=3 ./test_how2qa.sh diffrate-original-42.3
  ```
  Result
   - original-31.0: 75.68
   - original-34.7: 77.02
   - original-38.5: 81.24
   - original-42.3: 82.86
   - original:      85.27


## ReuseViT
### Extract Compressed Features
```
python -m dejavu.preprocess.extract_compressed msrvtt openai/clip-vit-base-patch16 1 train --dry-run
python -m dejavu.preprocess.extract_compressed msrvtt openai/clip-vit-base-patch16 1 train --extract-coded-order-only --dry-run
python -m dejavu.preprocess.extract_compressed msrvtt openai/clip-vit-base-patch16 2 test

python -m dejavu.preprocess.extract_compressed how2qa openai/clip-vit-large-patch14 2 frozenbilm --skip-pixel-check --use-start-end --dry-run
```

### Train ReuseViT
```
cd /workspace
./scripts/sacred/train.py /workspace/config/msrvtt/base.yaml
```
### Extract with train features
```
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages
cd /workspace
python -m dejavu.preprocess.extract_dataset --mode reuse --dataset msrvtt --reuse-model-name msrvtt/base --dry-run
```

### Measure Throughput
```
cd /workspace/profile
python run-nsys.py original --dataset msrvtt --batch-size 64
python run-nsys.py diffrate --dataset msrvtt --batch-size 64 --diffrate-target-flops 8.7 # {8.7, 10.0, 10.4, 11.5}
python run-nsys.py opt_attn --dataset msrvtt --batch-size 64 --reuse-model-name msrvtt/base
```