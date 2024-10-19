from ..clip import CLIPVisionModelWithProjection
import torch
from ...utils import PREDEFINED_PATHS

BASE_MODEL_NAME = 'openai/clip-vit-large-patch14'

class SimCacheLayer(torch.nn.Module):
    """
        Possible cache_key
            "input" : input
            "ln1"   : ln1(res1)
            "attn"  : attn(ln1)
            "res1"  : input + attn
            "ln2"   : ln2(res2)
            "mlp"   : mlp(ln2)
            "res2"  : mlp + res1
    """
    def __init__(self, original_encoder_layer, cache_key_list=["res1", "res2"]):
        super().__init__()
        self.original_encoder_layer = original_encoder_layer

        cache_key_candidates = ["input", "res1", "ln1", "attn", "res2", "ln2", "mlp"]
        for cache_key in cache_key_list:
            assert cache_key in cache_key_candidates, \
                f"cache_key should be one of {cache_key_candidates}"
        self.cache = {}
        for cache_target in cache_key_list:
            self.cache[cache_target] = None

    def _insert_to_cache_if_exist(self, key, tensor):
        if key in self.cache:
            self.cache[key] = [*tensor.clone().detach().split(1)]

    def forward(
            self,
            hidden_states,
            *args,
            output_attentions=False,
            output_qkvs=False,
            output_maps=False,
        ):
        B, N, dim = hidden_states.shape

        # Same as original encoder upto self attention
        residual = hidden_states
        self._insert_to_cache_if_exist("input", hidden_states)

        hidden_states = self.original_encoder_layer.layer_norm1(hidden_states)
        self._insert_to_cache_if_exist("ln1", hidden_states)

        hidden_states, attn_weights, qkvs = self.original_encoder_layer.self_attn(
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=True,
            output_qkvs=output_qkvs,
        )
        self._insert_to_cache_if_exist("attn", hidden_states)

        hidden_states = residual + hidden_states # [B, N, dim]
        residual = hidden_states
        self._insert_to_cache_if_exist("res1", hidden_states)

        hidden_states = self.original_encoder_layer.layer_norm2(hidden_states)
        self._insert_to_cache_if_exist("ln2", hidden_states)

        hidden_states = self.original_encoder_layer.mlp(hidden_states)
        self._insert_to_cache_if_exist("mlp", hidden_states)

        hidden_states = residual + hidden_states
        self._insert_to_cache_if_exist("res2", hidden_states)

        return hidden_states, 

    # input: rel_idxs: index which base zero. for example, if three indices are (10, 11, 12) -> (0, 1, 2)
    # returns cache dict
    def get_cache(self, rel_idxs):
        ret = {}
        for key in self.cache:
            ret[key] = [self.cache[key][idx] for idx in rel_idxs]
        return ret


def create_cache_model(base_model_name, cache_key_list=["res1", "res2"]):
    clip = CLIPVisionModelWithProjection.from_pretrained(
        base_model_name,
        cache_dir=PREDEFINED_PATHS['root']['cache'],
    )

    for layer_idx in range(len(clip.vision_model.encoder.layers)):
        original_encoder_layer = clip.vision_model.encoder.layers[layer_idx]
        clip.vision_model.encoder.layers[layer_idx] = SimCacheLayer(original_encoder_layer, cache_key_list=cache_key_list)
   
    return clip


def list_cache(cache_model, frame_rel_idxs):
    """
    returns: List[Dict[str, List[torch.Tensor]]]
        List: number of encoders
            Dict: key, List[torch.Tensor]
                List[torch.Tensor]: number of reference frames
    """
    return [l.get_cache(frame_rel_idxs) \
            for l in cache_model.vision_model.encoder.layers]


if __name__=="__main__":
    from torch.nn import functional as F
    key_list = ["input", "ln1", "attn", "res1", "ln2", "mlp", "res2"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_model = create_cache_model(BASE_MODEL_NAME, cache_key_list=key_list).to(device)

    random_input = torch.randn((3, 3, 224, 224)).to(device)
    ans_rel_idx = 0
    ref_rel_idxs = [1, 2]
    with torch.no_grad():
        out = cache_model(random_input)
    print(out.image_embeds.shape)
    ans_img_embeds = out.image_embeds[[ans_rel_idx]].clone().detach()
    cache = list_cache(cache_model, ref_rel_idxs)

    test_input = torch.randn((1, 3, 224, 224)).to(device)
    with torch.no_grad():
        test_out = cache_model(test_input)
    img_embeds = test_out.image_embeds

    print(ans_img_embeds.shape, img_embeds.shape)

    cos_sim = torch.sum(F.normalize(ans_img_embeds, dim=-1) * F.normalize(img_embeds, dim=-1), dim=-1)
    assert ans_img_embeds.shape == img_embeds.shape, "shape does not match"
    print(cos_sim)

    cos_sim = cos_sim.squeeze()
    cos_sim_history = [cos_sim]
    cos_sim_history = torch.stack(cos_sim_history)
    print(cos_sim_history)
    
    print(f"cache layer count : {len(cache)}")
    print(f"cache target : {cache[0].keys()}")

    for v in cache[0].values():
        print(f"num reference frames: {len(v)}")
        break

    for key in cache[0].keys():
        print(f"{key} cache tensor shape : {cache[0][key][0].shape}")
    """
    result:
        cache layer count : 24
        cache target : dict_keys(['input', 'ln1', 'attn', 'res1', 'ln2', 'mlp', 'res2'])
        num reference frames: 2
        input cache tensor shape : torch.Size([1, 257, 1024])
        ln1 cache tensor shape : torch.Size([1, 257, 1024])
        attn cache tensor shape : torch.Size([1, 257, 1024])
        res1 cache tensor shape : torch.Size([1, 257, 1024])
        ln2 cache tensor shape : torch.Size([1, 257, 1024])
        mlp cache tensor shape : torch.Size([1, 257, 1024])
        res2 cache tensor shape : torch.Size([1, 257, 1024])
    """
