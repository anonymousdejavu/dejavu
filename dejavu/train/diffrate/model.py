from ...model.diffrate import create_diffrate_model
from peft import LoraConfig, get_peft_model

from ...utils import get_diffrate_prune_merge


def DiffrateModel(
    base_model_name,
    dataset,
    diffrate_model_name,
    use_lora=False,
    lora_rank=4,
    lora_targets=["q_proj", "v_proj"]
):  
    prune_kept_nums, merge_kept_nums = get_diffrate_prune_merge(
        dataset,
        diffrate_model_name,
    )

    model = create_diffrate_model(
        base_model_name,
        prune_kept_nums,
        merge_kept_nums,
    )

    if use_lora:
        lora_config = LoraConfig(
            # From https://arxiv.org/pdf/2211.11733.pdf
            r=lora_rank,
            lora_alpha=4, # Set to same as r in original paper
            target_modules=lora_targets, 
            lora_dropout=0.1,
            bias="none"
        )

        model = get_peft_model(model, lora_config)

    return model

if __name__ == "__main__":
    BASE_MODEL_NAME = 'openai/clip-vit-base-patch16'

    model = DiffrateModel(
        base_model_name=BASE_MODEL_NAME,
        dataset='msrvtt',
        diffrate_model_name='original-10.4',
        use_lora=True,
    )

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
