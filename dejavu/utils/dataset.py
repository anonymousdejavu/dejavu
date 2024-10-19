#!/usr/bin/env python
import torch
from pathlib import Path
import numpy as np
import warnings

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from . import PREDEFINED_PATHS, rename_base_model
from .preprocess import is_integer, get_feature_path

# Filter and ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable.*")

def count_sampled_frames(video_path, sample_fps=1):
    import cv2
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


def sample_frames(video_path, sampling_fps=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return []

    # Get the frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize a counter for the time in seconds
    time_seconds = 0

    frames = []
    while True:
        # Calculate the frame number corresponding to the current time
        frame_number = round(time_seconds * frame_rate)

        # Break the loop if the frame number exceeds the total number of frames
        if frame_number >= total_frames:
            break

        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        # Increment the time counter by wanted sampling interval
        time_seconds += (1 / sampling_fps)

    # Release the video capture object
    cap.release()
    print(len(frames))
    return frames

def fill_list_to_length(a, num_frames):
    """
    Fill the list a to length 'frames' by repeating its elements.
    The elements from the end of the list are repeated more if necessary.
    """
    filled_list = []
    len_a = len(a)
    repeats = num_frames // len_a  # Number of times the whole list can be repeated
    additional_elements = num_frames % len_a  # Number of additional elements needed

    for i in range(len_a):
        # Repeat each element 'repeats' times or 'repeats + 1' times for the last few elements
        repeat_times = repeats + 1 if i >= len_a - additional_elements else repeats
        filled_list.extend([a[i]] * repeat_times)

    return filled_list


def make_best_indices(vid_len, sampling_fps=3, video_fps=30):
    indices = np.arange(0, vid_len, video_fps / sampling_fps, dtype=int)
    return indices


def sample_frames_with_gap(video_path, sampling_fps=3):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    # Get the frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = make_best_indices(total_frames, sampling_fps=sampling_fps, video_fps=video_fps)
    frames = []
    for frame_idx in frame_indices:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames


def sample_frames_all(video_path, frame_rate=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    print(len(frames))
    return frames

def sample_frames_CLIP4Clip(video_path, frame_rate=1):
    import cv2
    
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Samples a frame sample_fp X frames.
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration

    interval = 1
    if frame_rate > 0:
        interval = fps // frame_rate
    else:
        frame_rate = fps
    if interval == 0: interval = 1

    inds = [ind for ind in np.arange(0, fps, interval)]
    assert len(inds) >= frame_rate
    inds = inds[:frame_rate]

    ret = True
    frames, included = [], []

    for sec in np.arange(start_sec, end_sec + 1):
        if not ret: break
        sec_base = int(sec * fps)
        for ind in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

    cap.release()
    print(len(frames))
    return frames

def save_embedding(embedding, feature_path):
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    np.savez_compressed(feature_path, embeddings=embedding)

def load_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npz file.

    Args:
        feature_path (str): Path to the .npz file.

    Returns:
        embedding (np.ndarray): The feature vector.
    """
    with np.load(feature_path, allow_pickle=True) as data:
        embedding = data['embeddings']
    if return_pt:
        embedding = torch.from_numpy(embedding)
    return embedding

def load_npy_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npy file.

    Args:
        feature_path (str): Path to the .npy file.
        return_pt (bool): If True, returns a PyTorch tensor. Otherwise, returns a NumPy array.

    Returns:
        embedding (np.ndarray or torch.Tensor): The feature vector.
    """
    # Load the .npy file as a NumPy array
    embedding = np.load(feature_path)

    # Convert to PyTorch tensor if needed
    if return_pt:
        embedding = torch.from_numpy(embedding)

    return embedding

def get_all_frames(video_path):
    import cv2

    frames = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames

def fill_list_to_length(a, num_frames):
    """
    Fill the list a to length 'frames' by repeating its elements.
    The elements from the end of the list are repeated more if necessary.
    """
    filled_list = []
    len_a = len(a)
    repeats = num_frames // len_a  # Number of times the whole list can be repeated
    additional_elements = num_frames % len_a  # Number of additional elements needed

    for i in range(len_a):
        # Repeat each element 'repeats' times or 'repeats + 1' times for the last few elements
        repeat_times = repeats + 1 if i >= len_a - additional_elements else repeats
        filled_list.extend([a[i]] * repeat_times)

    return filled_list


def make_best_indices(vid_len, sampling_fps=3, video_fps=30):
    indices = np.arange(0, vid_len, video_fps / sampling_fps, dtype=int)
    return indices


def sample_frames_with_gap(video_path, sampling_fps=3):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    # Get the frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = make_best_indices(total_frames, sampling_fps=sampling_fps, video_fps=video_fps)
    frames = []
    for frame_idx in frame_indices:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Frame read successfully
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames


def sample_frames_all(video_path, frame_rate=1):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        # Read the frame
        ret, frame = cap.read()

        if not ret:
            break

        # Frame read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames

def sample_frames_CLIP4Clip(video_path, frame_rate=1):
    import cv2
    
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Samples a frame sample_fp X frames.
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration

    interval = 1
    if frame_rate > 0:
        interval = fps // frame_rate
    else:
        frame_rate = fps
    if interval == 0: interval = 1

    inds = [ind for ind in np.arange(0, fps, interval)]
    assert len(inds) >= frame_rate
    inds = inds[:frame_rate]

    ret = True
    frames, included = [], []

    for sec in np.arange(start_sec, end_sec + 1):
        if not ret: break
        sec_base = int(sec * fps)
        for ind in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

    cap.release()
    print(len(frames))
    return frames

def save_embedding(embedding, feature_path, mkdir=True):
    if mkdir:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    np.savez_compressed(feature_path, embeddings=embedding)

def load_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npz file.

    Args:
        feature_path (str): Path to the .npz file.

    Returns:
        embedding (np.ndarray): The feature vector.
    """
    with np.load(feature_path, allow_pickle=True) as data:
        embedding = data['embeddings']
    if return_pt:
        embedding = torch.from_numpy(embedding)
    return embedding

def load_npy_embedding(feature_path, return_pt=True):
    """
    Load a feature vector from a .npy file.

    Args:
        feature_path (str): Path to the .npy file.
        return_pt (bool): If True, returns a PyTorch tensor. Otherwise, returns a NumPy array.

    Returns:
        embedding (np.ndarray or torch.Tensor): The feature vector.
    """
    # Load the .npy file as a NumPy array
    embedding = np.load(feature_path)

    # Convert to PyTorch tensor if needed
    if return_pt:
        embedding = torch.from_numpy(embedding)

    return embedding

def get_feature_dir(dataset, base_model_name, fps, split, dir_key='feature_dir'):
    feature_root = Path(PREDEFINED_PATHS[dataset][dir_key])
    base_model_renamed = rename_base_model(base_model_name)

    # To refrain from naming file name 1.0
    if is_integer(fps):
        fps = int(fps)

    feature_dir = feature_root /  base_model_renamed / f'fps{fps}' / split
    return feature_dir

def get_feature_dir_malus03(dataset: str, pixel_or_compressed:str):
    assert dataset in ["msrvtt", "how2qa"]
    assert pixel_or_compressed in ["pixel", "compressed"]
    return Path(PREDEFINED_PATHS['malus03.throughput']['root_dir']) / dataset / pixel_or_compressed

def ray_get_with_tqdm(ret, num_results=None):
    import ray
    from tqdm import tqdm

    if num_results is None:
        num_results = len(ret)

    with tqdm(total=num_results) as pbar:
        while ret:
            # Get results as they complete
            done_refs, ret = ray.wait(ret)
            for ref in done_refs:
                pbar.update(1)
                ray.get(ref)

def count_available_features(
        feature_dir,
        video_ids,
        starts=None,
        ends=None,
        check_compressed=True,
        use_feature_path_v2=False,
    ):
    '''
    returns list in the format of (youtube_id, start, end, num_frames)
    '''

    ret = []

    for video_idx, video_id in enumerate(video_ids):
        if starts is not None:
            start = starts[video_idx]
            end = ends[video_idx]
        else:
            start = 0
            end = None

        frame_cnt = 0
        while end is None or start + frame_cnt <= end:
            pixel_path = get_feature_path(feature_dir, video_id, 'i', start + frame_cnt, use_v2=use_feature_path_v2)
            exists = pixel_path.exists()
            if check_compressed:
                compressed_path = get_feature_path(feature_dir, video_id, 'c', start + frame_cnt, use_v2=use_feature_path_v2)
                exists = exists and compressed_path.exists()
            if not exists:
                break
            frame_cnt += 1
        end = start + frame_cnt
        ret.append((video_id, start, end, frame_cnt))

    return ret