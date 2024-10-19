# from vit paper https://arxiv.org/pdf/2010.11929
# create a dataclass
import math
import numpy as np
from dataclasses import dataclass, field

precision = 4

@dataclass
class ViTConfig:
    model_type: str
    hidden_size: int = field(init=False)
    mlp_size: int = field(init=False)
    nlayer: int = field(init=False)
    nhead: int = field(init=False)
    npatch: int = field(init=False)

    def __post_init__(self):
        if self.model_type == "base":
            self.hidden_size = 768
            self.mlp_size = 3072
            self.nlayer = 12
            self.nhead = 12
            self.npatch = 14 ** 2 + 1
        elif self.model_type == "large":
            self.hidden_size = 1024
            self.mlp_size = 4096
            self.nlayer = 24
            self.nhead = 16
            self.npatch = 16 ** 2 + 1
        else:
            raise ValueError("model_type must be one of [base, large]")
    
    def nparams(self):
        return self.nlayer * (
            2 * 2 * (self.hidden_size)
            + (1 + self.hidden_size) * self.hidden_size * 3
            + (1 + self.hidden_size) * self.hidden_size
            + (1 + self.hidden_size) * self.mlp_size
            + (1 + self.mlp_size) * self.hidden_size
        )

# Setup flops
gelu_flop = 1
rsquare_flop = 1
exp_flop = 1

def layer_norm_1d_flops(dim: int):
    flops = 0
    # mean calculation
    # -- sum an divide
    flops += dim-1
    flops += 1
    # -- subtract mean
    flops += dim
    # std calculation
    # -- square
    flops += dim
    # -- sum squares
    flops += dim-1
    # -- mean square. mean calculated above / subtract (mean-square) - (square-mean) / root square / add eps
    flops += 2 + rsquare_flop + 1
    # -- divide std
    flops += dim
    # scale and shift --> this can be integrated with the next matmul
    # flops += 2 * dim
    return flops

def ln_flops(bs:int, dim: int):
    flops = 0
    # mean calculation
    # -- sum an divide
    flops += dim-1
    flops += 1
    # -- subtract mean
    flops += dim
    # std calculation
    # -- square
    flops += dim
    # -- sum squares
    flops += dim-1
    # -- mean square. mean calculated above / subtract (mean-square) - (square-mean) / root square / add eps
    flops += 2 + rsquare_flop + 1
    # -- divide std
    flops += dim
    # scale and shift --> this can be integrated with the next matmul
    # flops += 2 * dim
    return bs * flops

def mm_flops(m:int, k:int, n:int):
    # flops of dot-product for single output element
    flops = k + k-1
    
    # scale it to flops of entire output matrix
    flops *= (m * n)

    return flops

def softmax_1d_flops(dim:int):
    flops = 0
    # numerical stability
    # -- get max
    flops += dim-1
    # -- subtract max value
    flops += dim
    # exponentiation
    flops += exp_flop * dim
    # sum exponentiation
    flops += dim-1
    # divide
    flops += dim
    return flops

def softmax_flops(bs:int, dim:int):
    flops = 0
    # numerical stability
    # -- get max
    flops += dim-1
    # -- subtract max value
    flops += dim
    # exponentiation
    flops += exp_flop * dim
    # sum exponentiation
    flops += dim-1
    # divide
    flops += dim
    return bs * flops

def gelu_flops(bs:int):
    return bs * gelu_flop

def vnorm_flops(bs:int, dim:int):
    flops = 0
    # square / sum / root square / eps
    flops += dim + dim - 1 + rsquare_flop + 1
    # divide all
    flops += dim
    return bs * flops

def cumsum_flops(bs:int, dim:int):
    flops = 0
    # refer to https://en.wikipedia.org/wiki/Prefix_sum, Shorter span, more parallel
    return bs * (dim * (math.log2(dim) - 1) + 1)

class NaiveVitFlopsLogger:
    def __init__(self, model_config: ViTConfig, batch_size):
        self.model_config = model_config
        self.batch_size = batch_size # Batch
        self.latent_vector_size = self.model_config.hidden_size # D
        self.num_heads = self.model_config.nhead
        self.num_encoder_layers = self.model_config.nlayer
        self.num_tokens = self.model_config.npatch # N, this include class token
        self.mlp_hidden_size = self.model_config.mlp_size

        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_total = 0

    def reset_flops(self):
        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_total = 0

    def sum_flops(self):
        self.flops_total = self.flops_qkv_projection + self.flops_attention + self.flops_ffn

    def get_flops(self):
        self.reset_flops()

        # transformer encoder layers
        for i in range(self.num_encoder_layers):
            self.set_flops_of_encoder(i)

        # final output project (reference: CLIP)
        # single linear layer with a class token
        self.flops_ffn += self.batch_size * mm_flops(m=1,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)
        
        self.sum_flops()
        
        return self.flops_total

    def set_flops_of_encoder(self, i=None):
        # 1st layer norm
        self.flops_qkv_projection += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

        # MHA, output dimension: [Batch, N, D]
        self.set_flops_of_multi_head_attention()

        num_tokens = 1 if i == self.num_encoder_layers - 1 else self.num_tokens

        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)

        # 1st residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size

        # 2nd layer norm
        # self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        self.flops_ffn += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        
        # MLP
        self.set_flops_of_mlp(num_tokens)
        
        # 2nd residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size

    def set_flops_of_multi_head_attention(self):
        # qkv linear
        # batch_size x [N, D] x [D, D]
        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=self.num_tokens,
                                                                    k=self.latent_vector_size,
                                                                    n=self.latent_vector_size)
        
        # MHA
        assert self.latent_vector_size % self.num_heads == 0
        latent_vector_size_per_head = int(self.latent_vector_size / self.num_heads)

        # attention scores
        # Aw = q x k^t, output dimentions: [batch_size, num_heads, num_tokens, num_tokens]
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=latent_vector_size_per_head,
                                                                            n=self.num_tokens)

        # scaling
        # Aw = Aw / scale
        # we don't consider flops of calculating the scaling constant (it is negligible)
        # TODO: previous submission
        # self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2)
        self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * self.num_tokens * softmax_1d_flops(dim=self.num_tokens)

        # attention outputs
        # A = Aw x v
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=self.num_tokens,
                                                                            n=latent_vector_size_per_head)

        # linear layer
        # print(self.batch_size * mm_flops(m=self.num_tokens, k=self.latent_vector_size, n=self.latent_vector_size))
        # self.flops_attention += self.batch_size * mm_flops(m=self.num_tokens,
        #                                                    k=self.latent_vector_size,
        #                                                    n=self.latent_vector_size)

    def set_flops_of_mlp(self, num_tokens):
        # 1st linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)

        # gelu
        self.flops_ffn += self.batch_size * num_tokens * self.mlp_hidden_size

        # 2nd
        # TODO: previous submission
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)
        # self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
        #                                              k=self.mlp_hidden_size,
        #                                              n=self.latent_vector_size)
class EventfulTransformerFlopsLogger(NaiveVitFlopsLogger):
    def __init__(self, model_config: ViTConfig, batch_size, r):
        super().__init__(model_config, batch_size)

        # number of tokens to reuse
        self.r = r
        # print(f"tokens: {self.num_tokens}")

        assert self.r <= self.num_tokens

        self.flops_gating_module = 0
        self.flops_delta_gating_module = 0

    def reset_flops(self):
        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_total = 0
        self.flops_gating_module = 0
        self.flops_delta_gating_module = 0

    def sum_flops(self):
        self.flops_total = self.flops_qkv_projection + self.flops_attention + self.flops_ffn + \
            self.flops_gating_module + self.flops_delta_gating_module
        
    def set_flops_of_encoder(self, i=None):
        # 1st layer norm
        self.flops_qkv_projection += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

        # MHA, output dimension: [Batch, N, D]
        self.set_flops_of_multi_head_attention()

        num_tokens = 1 if i == self.num_encoder_layers - 1 else self.r

        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)

        # 1st residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size

        # 2nd layer norm
        # self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        self.flops_ffn += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        # self.flops_ffn += self.batch_size * self.r * layer_norm_1d_flops(dim=self.latent_vector_size)
        
        # MLP
        self.set_flops_of_mlp(num_tokens)
        
        # 2nd residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size


    def set_flops_of_multi_head_attention(self):
        # gating module
        # q, k, v -> q', k', v' ([N, D] -> [M, D] / N: num_tokens, M: r)
        # self.flops_gating_module +=  self.batch_size * self.get_flops_of_gating_module(num_tokens=self.num_tokens,
        #                                                                                latent_vector_size=self.latent_vector_size)
        self.flops_gating_module +=  self.batch_size * self.get_flops_of_gating_module(num_tokens=self.r,
                                                                                       latent_vector_size=self.latent_vector_size)


        # qkv linear
        # batch_size x [N, D] x [D, D]
        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=self.r,
                                                                    k=self.latent_vector_size,
                                                                    n=self.latent_vector_size)
        
        # MHA
        assert self.latent_vector_size % self.num_heads == 0
        latent_vector_size_per_head = int(self.latent_vector_size / self.num_heads)

        # attention scores
        # Aw = q x k^t, output dimentions: [batch_size, num_heads, num_tokens, num_tokens]
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=latent_vector_size_per_head,
                                                                            n=self.num_tokens)
        
        # scaling
        # Aw = Aw / scale
        # we don't consider flops of calculating the scaling constant (it is negligible)
        # TODO: previous submission
        self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * self.num_tokens * softmax_1d_flops(dim=self.num_tokens)

        # delta gating module
        self.flops_delta_gating_module += self.batch_size * self.num_heads * self.get_flops_of_delta_gating_module(num_tokens=self.r,
                                                                                                                   latent_vector_size=latent_vector_size_per_head)
        self.flops_delta_gating_module += self.batch_size * self.num_heads * self.get_flops_of_delta_gating_module(num_tokens=self.r,
                                                                                                                   latent_vector_size=self.num_tokens)
        # attention outputs
        # A = Aw' x v'
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=self.num_tokens,
                                                                            n=latent_vector_size_per_head)

        # # prev_Aw x delta_v
        # self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
        #                                                                     k=self.r,
        #                                                                     n=latent_vector_size_per_head)
        
        # # prev_delta_A x (prev_v - delta_v)
        # self.flops_attention += self.batch_size * self.num_heads * (self.r * latent_vector_size_per_head)
        # self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
        #                                                                     k=self.r,
        #                                                                     n=latent_vector_size_per_head)

        # # prev_attention_output + (prev_Aw x delta_v) + (prev_delta_A x (prev_v - delta_v))
        # self.flops_attention += 2 * (self.batch_size * self.num_heads * (self.num_tokens * latent_vector_size_per_head))

        # gating module
        self.flops_gating_module += self.get_flops_of_gating_module(num_tokens=self.r,
                                                                    latent_vector_size=self.latent_vector_size)

        # linear layer
        # self.flops_attention += self.batch_size * mm_flops(m=self.r,
        #                                                    k=self.latent_vector_size,
        #                                                    n=self.latent_vector_size)

    def set_flops_of_mlp(self, num_tokens):
        # gating module
        self.flops_gating_module += self.get_flops_of_gating_module(num_tokens=num_tokens,
                                                                    latent_vector_size=self.latent_vector_size)

        # 1st linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)

        # gelu
        self.flops_ffn += self.batch_size * self.r * self.mlp_hidden_size

        # 2nd linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.mlp_hidden_size,
                                                     n=self.latent_vector_size)
    
    def get_flops_of_gating_module(self, num_tokens, latent_vector_size):
        # cacluate e = c - u
        flops = num_tokens * latent_vector_size

        # calculate
        for _ in range(num_tokens):
            # elementwise power(D, 2)
            flops += latent_vector_size

            # reduce
            flops += latent_vector_size - 1

            # sqrt
            flops += 1

        # select r tokens
        flops += int(math.ceil(num_tokens * math.log2(num_tokens) + self.r))

        return flops
    
    def get_flops_of_delta_gating_module(self, num_tokens, latent_vector_size):
        return self.get_flops_of_gating_module(num_tokens=num_tokens,
                                               latent_vector_size=latent_vector_size)

# TODO[Yoonsung]: implement this
class CmcFlopsLogger(NaiveVitFlopsLogger):
    def __init__(self, model_config: ViTConfig, batch_size, avg_reuse_ratio):
        super().__init__(model_config, batch_size)

        self.avg_reuse_ratio = avg_reuse_ratio

    def set_flops_of_multi_head_attention(self):
        # qkv linear
        # batch_size x [N, D] x [D, D]
        # CMC only apply the reuse computation on QKV projection
        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=int(self.num_tokens * (1. - self.avg_reuse_ratio)),
                                                                    k=self.latent_vector_size,
                                                                    n=self.latent_vector_size)
        
        # MHA
        assert self.latent_vector_size % self.num_heads == 0
        latent_vector_size_per_head = int(self.latent_vector_size / self.num_heads)

        # attention scores
        # Aw = q x k^t, output dimentions: [batch_size, num_heads, num_tokens, num_tokens]
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=latent_vector_size_per_head,
                                                                            n=self.num_tokens)

        # scaling
        # Aw = Aw / scale
        # we don't consider flops of calculating the scaling constant (it is negligible)
        self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * self.num_tokens * softmax_1d_flops(dim=self.num_tokens)

        # attention outputs
        # A = Aw x v
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=self.num_tokens,
                                                                            n=latent_vector_size_per_head)

        # linear layer
        # self.flops_attention += self.batch_size * mm_flops(m=self.num_tokens,
        #                                                    k=self.latent_vector_size,
        #                                                    n=self.latent_vector_size)
@dataclass
class RunConfig:
    model_config: ViTConfig
    batch_size: int
    reuse_rates_gen: callable
    # [layer_idx][iter_idx]
    reuse_rates: list = field(init=False)
    
    def __post_init__(self):
        self.reuse_rates = self.reuse_rates_gen(self.model_config)

class MemoryLogger:
    def __init__(self, model_config: ViTConfig, run_config: RunConfig):
        self.mcfg = model_config
        self.rcfg= run_config
        
        self.init_params()
        self.model_size = self.mcfg.nparams()

        # this is for cache
        self.layers = [[
            'stage_states.1',
            'stage_states.2',
            'stage_states.3',
            'stage_states.4',
        ]
        for i in range(self.mcfg.nlayer)]

        # this is for flops calculation
        self.layers = [[
            'ln1', # done
            'qkvgen', # done
            'restoration' if i > 0 else None, # todo: it is restoration now
            'mha', # done

            # 'cls_attn_sum',
            
            # - stage states
            'cls_sum' if i < self.mcfg.nlayer - 1 else None,
            'vnorm1' if i < self.mcfg.nlayer - 1 else None,
            'vnorm2' if i < self.mcfg.nlayer - 1 else None,
            # -- iter 4
            'stage_states.1',
            'stage_states.2',
            'stage_states.3',
            'stage_states.4',

            'proj', # done
            'res1', # done
            'ln2', # done
            'fc1', # done
            'gelu', # done        
            'fc2', # done
            'res2', # done
        ]
        for i in range(self.mcfg.nlayer)]
    
    def log_filtered(self, name_filter: callable):
        self.init_params()
        logs = []
        for layer_idx, layer_names in enumerate(self.layers):
            for layer_name in layer_names:
                if layer_name is None:
                    continue
                self.update_params(layer_idx, layer_name)
                if name_filter(layer_name):
                    logs.append(self.log(f"layer{layer_idx}.{layer_name}"))
        
        return logs

    def log_cache_size(self):
        def logic(layer_name:str):
            return layer_name.startswith('stage_states')
        return self.log_filtered(logic)
    
    def init_params(self):
        # total 1 cache track. only consider ref_norm for 11 layers
        # hqkv is tracked as activation.
        self.cache_size = self.rcfg.batch_size * (self.mcfg.nlayer-1) * (self.mcfg.npatch-1) * self.mcfg.hidden_size
        # self.cache_size = self.rcfg.batch_size * (self.mcfg.npatch-1) * self.mcfg.hidden_size
        # 4 frames each
        self.activation_size = 4 * self.rcfg.batch_size * self.mcfg.npatch * self.mcfg.hidden_size 
        self.flops = 0
        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_overhead = 0
    
    def log_all(self, is_flops_per_layer: bool = False):
        logs = []
        for layer_idx, layer_names in enumerate(self.layers):
            for layer_name in layer_names:
                if layer_name is None:
                    continue
                self.update_params(layer_idx, layer_name)
                logs.append(self.log(f"layer{layer_idx}.{layer_name}"))
                if is_flops_per_layer:
                    self.flops = 0
        
        return logs
    
    def get_flops(self):
        return self.flops
    
    def log(self, label):
        return {
            "label": label,
            "cache_size": self.cache_size * precision / 1024 / 1024 / 1024, # precision & to megabytes
            "activation_size": self.activation_size * precision / 1024 / 1024 / 1024,
            "model_size": self.model_size * precision / 1024 / 1024 / 1024,
            "flops": self.flops,
        }
        
    # layer_idx is 0 ~ nlayer - 1
    def update_params(self, layer_idx:int, layer_name:str):
        
        if layer_idx < self.mcfg.nlayer - 1:
            reuse_rate_sum = sum(self.rcfg.reuse_rates[layer_idx])
        if layer_idx > 0:
            qkv_reuse_rate_sum = sum(self.rcfg.reuse_rates[layer_idx-1])
        
        if layer_name in ['ln1', 'ln2']:
            flops = ln_flops(self.rcfg.batch_size * self.mcfg.npatch, self.mcfg.hidden_size)

            if layer_name == 'ln1':
                self.flops_qkv_projection += flops
            else:
                self.flops_ffn += flops

            self.flops += flops
        elif layer_name in ['qkvgen', 'proj', 'fc1', 'fc2']:
            if layer_name == 'qkvgen':
                if layer_idx == 0:
                    bs = 4 * self.rcfg.batch_size * self.mcfg.npatch
                elif layer_idx > 0:
                    bs = (4 - qkv_reuse_rate_sum) * self.rcfg.batch_size * self.mcfg.npatch
            elif layer_name in ['proj', 'fc1', 'fc2']:
                if layer_idx < self.mcfg.nlayer - 1:
                    bs = (4 - reuse_rate_sum) * self.rcfg.batch_size * self.mcfg.npatch
                elif layer_idx == self.mcfg.nlayer - 1:
                    bs = 4 * self.rcfg.batch_size

            if layer_name in ['qkvgen', 'proj', 'fc1']:
                dim1 = self.mcfg.hidden_size
            elif layer_name == 'fc2':
                dim1 = self.mcfg.mlp_size

            if layer_name == 'qkvgen':
                dim2 = 3 * self.mcfg.hidden_size 
            elif layer_name == 'fc1':
                dim2 = self.mcfg.mlp_size
            elif layer_name in ['proj', 'fc2']:
                dim2 = self.mcfg.hidden_size

            flops = mm_flops(bs, dim1, dim2)

            if layer_name == 'qkvgen':
                self.flops_qkv_projection += flops
            else:
                self.flops_ffn += flops
            self.flops += flops

            self.activation_size = bs * dim2
        
        elif layer_name in ['res1', 'res2']:
            self.flops_ffn += self.activation_size
            self.flops += self.activation_size
        
        elif layer_name == 'gelu':
            flops = gelu_flops(self.activation_size)
            self.flops_ffn += flops
            self.flops += flops
        
        elif layer_name == 'mha':
            bs = self.rcfg.batch_size
            nh = self.mcfg.nhead
            hph = self.mcfg.hidden_size // nh # hidden per head
            flops = 0

            # mm1 batchsize*numhead 개의 (npatch, hph) @ (hph, npatch)
            flops += mm_flops(bs * nh * self.mcfg.npatch, hph, self.mcfg.npatch)
            # (bs, nh, npatch, npatch) scale, softmax
            flops += bs * nh * self.mcfg.npatch ** 2
            flops += softmax_flops(bs * nh * self.mcfg.npatch, self.mcfg.npatch)
            # mm2
            flops += mm_flops(bs * nh * self.mcfg.npatch, self.mcfg.npatch, hph)

            flops = 4 * flops

            self.flops_attention += flops
            self.flops += flops

        elif layer_name == 'restoration':
            if self.mcfg.nlayer == 0:
                return
            # mlp * 4 / for parameter, please refer to dejavu/model/train/restoration.py#43
            inner_dim = 64
            bs = 4 * self.rcfg.batch_size * self.mcfg.npatch
            flops = 4 * (mm_flops(bs, self.mcfg.hidden_size, inner_dim) + bs * inner_dim + mm_flops(bs, inner_dim, self.mcfg.hidden_size)) # todo         
            self.flops_overhead += flops
            self.flops += flops
        
        elif layer_name == 'cls_sum':
            if layer_idx == self.mcfg.nlayer - 1:
                return
            flops = (self.mcfg.nhead - 1) * 4 * self.rcfg.batch_size * (self.mcfg.npatch - 1)
            self.flops_overhead += flops
            self.flops += flops
        elif layer_name in ['vnorm1', 'vnorm2']:
            if layer_idx == self.mcfg.nlayer - 1:
                return
            # dejavu/model/reuse/opt_attention/model.py #316,317
            flops = vnorm_flops(9 * self.rcfg.batch_size * self.mcfg.npatch, self.mcfg.hidden_size)
            self.flops_overhead += flops
            self.flops += flops

        elif layer_name in ['stage_states.1', 'stage_states.2', 'stage_states.3', 'stage_states.4']:
            if layer_idx == self.mcfg.nlayer - 1:
                return
            iter_idx = int(layer_name[-1:])-1
            reuse_rate = self.rcfg.reuse_rates[layer_idx]

            flops = 0

            flops += self.rcfg.batch_size * self.mcfg.npatch * mm_flops(1, self.mcfg.hidden_size, 1)
            if iter_idx >= 1:
                flops += self.rcfg.batch_size * self.mcfg.npatch * mm_flops(1, self.mcfg.hidden_size, 1)
                flops += self.rcfg.batch_size * self.mcfg.npatch

            #329 forward inner / refer to variables at dejavu/model/train/decision.py#68
            # assume using compressed info
            input_dim = 3
            inner_dim = 64
            # bn(input_dim) --> bn can be fused with future matmul
            # mlp
            flops += self.rcfg.batch_size * (mm_flops(self.mcfg.npatch-1, input_dim, inner_dim) + (self.mcfg.npatch-1) * inner_dim + mm_flops(self.mcfg.npatch-1, inner_dim, 1) )
            
            #335 decision > 0
            flops += self.rcfg.batch_size * self.mcfg.npatch

            #348 stage_states_per_batch
            # ~reuse_map
            flops += self.mcfg.npatch * self.rcfg.batch_size
            # torch.cumsum(reuse_map)
            flops += cumsum_flops(self.rcfg.batch_size, self.mcfg.npatch)

            # compute_indices = cumsum - 1
            flops += self.rcfg.batch_size * self.mcfg.npatch
            # torch.cumsum(compute_cnts)
            flops += cumsum_flops(1, self.rcfg.batch_size)

            # pre_proj - most_similar_pre_proj
            flops += self.rcfg.batch_size * self.mcfg.npatch * self.mcfg.hidden_size
            # reference_cache_len += compute_cnts
            flops += 1
            # compute_cache_len += compute_total
            flops += 1

            self.flops_overhead += flops
            self.flops += flops
            
            if iter_idx < 2:
                self.cache_size += self.rcfg.batch_size * (self.mcfg.npatch - 1) * self.mcfg.hidden_size
            else:
                self.cache_size -= self.rcfg.batch_size * (self.mcfg.npatch - 1) * self.mcfg.hidden_size

        else:
            raise ValueError(f"unknown layer name {layer_name}")

def get_reusevit_flops(model_size, reuse_rate, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    assert len(reuse_rate) == 4, "reuse rate must be a list of 4 elements"

    model_config = ViTConfig(model_size)
    run_config = RunConfig(model_config, batch_size, lambda model_config: [reuse_rate] * (model_config.nlayer -1))
    memory_logger = MemoryLogger(model_config, run_config)
    logs_ours = memory_logger.log_cache_size()
    # print(logs_ours)
    flops = memory_logger.get_flops()

    return int(flops / 4)

def get_orginal_vit_flops(model_size, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    naive_vit_flops_logger = NaiveVitFlopsLogger(model_config=model_config, batch_size=batch_size) 
    return naive_vit_flops_logger.get_flops()

def get_eventful_flops(model_size, r, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    eventful_transformer_flops_logger = EventfulTransformerFlopsLogger(model_config=model_config,
                                                                   batch_size=batch_size,
                                                                   r=r)
    return eventful_transformer_flops_logger.get_flops()

def get_cmc_flops(model_size, avg_reuse_ratio, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    cmc_flops_logger = CmcFlopsLogger(model_config=model_config,
                                     batch_size=batch_size,
                                     avg_reuse_ratio=avg_reuse_ratio)
    return cmc_flops_logger.get_flops()