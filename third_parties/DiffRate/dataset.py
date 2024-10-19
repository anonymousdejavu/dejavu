# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from pathlib import Path
import cv2
from joblib import dump, load
import numpy as np

import sys
if '/workspace' not in sys.path:
    sys.path.append('/workspace')
from dejavu.dataset import How2qaDataset, MsrvttDataset


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def count_sampled_frames(video_path, sample_fps=1):
    """
    Count the number of frames that would be sampled from a video.

    Args:
        video_path (str): Path to the video file.
        sample_fps (int): The number of frames to sample per second.

    Returns:
        int: Number of frames that would be sampled.
    """
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # If FPS is not available, count frames manually
    if not video_fps:
        total_frames = 0
        while True:
            # Try to read a frame
            has_frame, _ = cap.read()
            if has_frame:
                total_frames += 1
            else:
                break
    else:
        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Close the video file
    cap.release()

    # Calculate video duration in seconds
    video_duration = total_frames / video_fps if video_fps else total_frames
    # Calculate number of frames to be sampled
    sampled_frames = int(video_duration * sample_fps)

    return sampled_frames

def sample_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    # Calculate the frame interval based on the desired frame rate
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / frame_rate)
    frame_num = 0
    frames = []
    # Iterate through the frames until we reach the desired frame rate
    while True:
        if frame_num % frame_interval != 0:
            ret = cap.grab()
            # Check if the frame was successfully read
            if not ret:
                break
            frame_num += 1
        else:
            # Read the frame
            ret, frame = cap.read()
            # Check if the frame was successfully read
            if not ret:
                break
            frame_num += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames

# Build dataset from the frames
class CharadesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            base_model_name,
            dataset_dir,
            train=True,
            num_video=-1,
            skip_dump=True,
            use_cache=True,
            device='cpu',
            dataset_name='charades',
        ):
        
        CACHE_DIR='/mnt/ssd1/cache'

        if base_model_name == 'vit_base_patch16_clip_224.openai':
            base_model_name = 'openai/clip-vit-base-patch16'
        elif base_model_name == 'vit_large_patch14_clip_224.openai':
            base_model_name = 'openai/clip-vit-large-patch14'
        else:
            raise NotImplementedError(f"Unknown model name {base_model_name}")

        if train:
            split = 'train'
        else:
            split = 'test'

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        dataset = load_dataset(
            "HuggingFaceM4/charades",
            name='480p',
            split=split,
            cache_dir=CACHE_DIR,
        )
        if num_video == -1:
            num_video = len(dataset)
        self.num_video = num_video
        self.skip_dump = skip_dump
        self.use_cache = use_cache
        print(f'Loading {num_video} videos')
        
        self.model = CLIPModel.from_pretrained(
            base_model_name,
            cache_dir=CACHE_DIR
        ).to(self.device)
        self.nb_classes = self.model.config.vision_config.projection_dim
        self.processor = CLIPProcessor.from_pretrained(base_model_name, cache_dir=CACHE_DIR)

        base_model_name_renamed = base_model_name.replace('/', '_')
        self.cache_file_dir = Path(dataset_dir) / dataset_name / base_model_name_renamed
        self.cache_file_dir.mkdir(parents=True, exist_ok=True)

        dataset_info_path = self.cache_file_dir.parent / f'{split}_dataset_info.pkl'
        # Initiation takes too long, so we dump the dataset info 
        if dataset_info_path.exists():
            print(f"Reusing {dataset_info_path}")
            self.video_paths, self.idx_to_video, self.video_to_start, self.len = load(dataset_info_path)
        else:
            print(f"Populating {dataset_info_path}")
            self.len = 0
            self.video_paths = []
            self.idx_to_video = []
            self.video_to_start = []

            for video_idx in range(num_video):
                video_path_str = dataset[video_idx]['video']
                video_path = Path(video_path_str)

                num_frames = count_sampled_frames(video_path)
                self.video_paths.append(video_path)
                sample_cnt = num_frames - 1
                for _ in range(sample_cnt):
                    self.idx_to_video.append(video_idx)
                self.video_to_start.append(self.len)
                self.len += sample_cnt

            # Dump the dataset info only when wer are using all the videos
            if num_video == len(dataset):
                print("Saving dataset info to", dataset_info_path)
                dump((self.video_paths, self.idx_to_video, self.video_to_start, self.len), dataset_info_path)

        print("Total number of frames", self.len)

        if num_video != len(self.video_paths):
            self.len = self.video_to_start[num_video]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Find the video index
        video_idx = self.idx_to_video[idx]
        start_idx = self.video_to_start[video_idx]

        frame_idx = idx - start_idx

        video_path = self.video_paths[video_idx]
        video_path = Path(video_path)

        p_path = self.cache_file_dir / f'{video_path.stem}_p_{frame_idx}.npz'
        o_path = self.cache_file_dir / f'{video_path.stem}_o_{frame_idx}.npz'

        # Try to load from file
        loaded_from_file = False
        if all([self.use_cache, p_path.exists(), o_path.exists()]):
            try:
                with np.load(p_path, allow_pickle=True) as data:
                    numpy_array = data['embeddings']
                pixel_values = torch.tensor(numpy_array)

                with np.load(o_path, allow_pickle=True) as data:
                    numpy_array = data['embeddings']
                output = torch.tensor(numpy_array)

                pixel_values = pixel_values.to(torch.float32)
                output = output.to(torch.float32)

                # pixel_values = load(p_path)
                # output = load(o_path)
                loaded_from_file = True
            except Exception as e:
                print("Error loading from file", str(e))
                pass

        if not loaded_from_file:
            print("Extracting features from video", video_path)
            frames = sample_frames(video_path)

            pixel_values = self.processor.image_processor(
                images=frames, return_tensors="pt", padding=True, 
            )["pixel_values"]

            pixel_values_device = pixel_values.to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(pixel_values=pixel_values_device).to('cpu')

            # Save to file
            if not self.skip_dump:
                for i, (p, o) in enumerate(zip(pixel_values, outputs)):
                    dump(p, self.cache_file_dir / f'{video_path.stem}_p_{i}.pkl', compress=3)
                    dump(o, self.cache_file_dir / f'{video_path.stem}_o_{i}.pkl', compress=3)
                    # torch.save(p, self.cache_file_dir / f'{video_path.stem}_p_{i}.pt')
                    # torch.save(o, self.cache_file_dir / f'{video_path.stem}_o_{i}.pt')

            # pixel_values = pixel_values[frame_idx]
            pixel_values = self.processor.image_processor(
                images=frames, return_tensors="pt", padding=True, 
                # do_rescale=False,
                # do_normalize=False,
                # do_convert_rgb=False,
            )["pixel_values"].float()[frame_idx]
            output = outputs[frame_idx]

        return pixel_values, output

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CHARADES':
        dataset = CharadesDataset(args.model, dataset_dir=args.data_path, train=is_train)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'HOW2QA':
        if args.model == 'vit_large_patch14_clip_224.openai':
            base_model_name = 'openai/clip-vit-large-patch14'
            nb_classes = 768
        # elif args.model == 'vit_base_patch16_clip_224.openai':
        #     base_model_name = 'openai/clip-vit-base-patch16'
        #     nb_classes = 512
        else:
            raise NotImplementedError(f"Unknown model name {args.model}")
        dataset = How2qaDataset(
            # split='train' if is_train else 'test',
            split='test' if is_train else 'frozenbilm',
            base_model_name=base_model_name,
            fps=2,
        )
    elif args.data_set == 'MSRVTT':
        if args.model == 'vit_base_patch16_clip_224.openai':
            base_model_name = 'openai/clip-vit-base-patch16'
            nb_classes = 512
        else:
            raise NotImplementedError(f"Unknown model name {args.model}")
        dataset = MsrvttDataset(
            split='train' if is_train else 'test',
            base_model_name=base_model_name,
            fps=2,
        )
    else:
        raise NotImplementedError(f"Unknown dataset {args.data_set}")

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

# Test if huggingface weight matches that of timm
if __name__ == '__main__':
    model_name = 'vit_base_patch16_clip_224.openai'
    # model_name = 'vit_large_patch14_clip_224.openai'

    from timm import create_model
    class QuickGELU(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            return x * torch.sigmoid(1.702 * x)

    model = create_model(
        model_name,
        pretrained=True,
        # pretrained_cfg_overlay={
        #     'mean': 0.,
        #     'std': 1.,
        #     'crop_pct': 1.0,
        # },
        act_layer=QuickGELU,
    )
    print(model.pretrained_cfg)
    dataset_dir = '/mnt/ssd2/dataset'
    # Not using cache to test if the dataset is correct
    # ds = CharadesDataset(model_name, dataset_dir=dataset_dir, train=True, use_cache=False)
    ds = CharadesDataset(model_name, dataset_dir=dataset_dir, train=False)

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # for i in range(10):
    #     p, o = ds[i]
    #     print('Shapes:', p.shape, o.shape)
    #     with torch.no_grad():
    #         timm_o = model(p.unsqueeze(0))[0]

    #     mse = (o - timm_o).pow(2).mean()
    #     sim = cosine_similarity(o, timm_o).mean()
    #     print(f"MSE: {mse.item():.4f}, Cosine Similarity: {sim.item():.4f}")


    from DiffRate.patch import clip
    clip(model)
    all_kept_num = [197] * len(model.blocks)
    model.eval()
    model.set_kept_num(all_kept_num, all_kept_num)

    for i in range(10):
        p, o = ds[i]
        print('Shapes:', p.shape, o.shape)
        with torch.no_grad():
            timm_o = model(p.unsqueeze(0))[0]

        mse = (o - timm_o).pow(2).mean()
        sim = cosine_similarity(o, timm_o).mean()
        print(f"MSE: {mse.item():.4f}, Cosine Similarity: {sim.item():.4f}")


