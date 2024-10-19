from pathlib import Path

def rename_base_model(base_model_name):
    base_model_renamed = base_model_name.replace('/', '_')
    return base_model_renamed

def parse_lora_targets(lora_targets):
    input_list = lora_targets.split(',')
    result_list = [item + '_proj' for item in input_list]
    return result_list

def get_cropped_video_path(video_path, start, end):
    video_path = Path(video_path)
    return video_path.parent / f'{video_path.stem}_{start}_{end}.mp4'