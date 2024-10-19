import torch
from torch import nn
from ...utils.train import init_threshold, get_monotonic_threshold

class ReuseThreshold(nn.Module):
    def __init__(
            self, 
            kept_num, 
            threshold_act='',
            disable_monotonicity=False,
            use_compressed_info=False,
            use_tanh=False,
            disable_final_tanh=False,
            decision_hyperparam=None,
        ):
        super().__init__()
        # self.sim_threshold = nn.Parameter(torch.zeros((kept_num,)), requires_grad=True)
        self.act = threshold_act
        self.disable_monotonicity = disable_monotonicity
        self.use_compressed_info = use_compressed_info
        
        self.sim_threshold = nn.Parameter(torch.Tensor(kept_num))
        init_threshold(self.sim_threshold, threshold_act, disable_monotonicity)

        self.use_tanh = use_tanh
        if decision_hyperparam is not None:
            decision_hyperparam = nn.Parameter(torch.Tensor([decision_hyperparam]))
        self.decision_hyperparam = decision_hyperparam

    def forward(self, importance, similarity, compressed_map, **kwargs):
        # [B, current_token, cached_token] => [B, current_token]
        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        if self.decision_hyperparam is None:
            threshold = get_monotonic_threshold(
                self.sim_threshold,
                act=self.act,
                disable_monotonicity=self.disable_monotonicity
            )
        else:
            threshold = self.decision_hyperparam

        if importance is not None:
            importance = importance.sum(dim=1)      # [B, N-1]
            importance = torch.argsort(importance, dim=-1, descending=True)
            rank = torch.argsort(importance, dim=-1, descending=False) # [B, N-1]

            # When similarity is lower than threshold, use newly computed value
            B, _ = rank.shape
            threshold = torch.gather(
                threshold.unsqueeze(0).expand(B, -1),
                dim=1,
                index=rank
            )

        if compressed_map is not None:
            compressed_map = compressed_map.squeeze(1)
            threshold = threshold - compressed_map

        reuse_decision = most_similar_score - threshold

        reuse_decision = reuse_decision.unsqueeze(-1)

        return reuse_decision, most_similar_idx, threshold



class ReuseMLP(nn.Module):
    def __init__(
            self,
            inner_dim=64,
            layer_pattern=None,
            use_compressed_info=False,
            decision_mlp_disable_bias=False,
            decision_mlp_use_norm=False,
            decision_initialize=None,
            decision_mlp_out_dim=1,
            decision_reference_type=False,
            dropout=0.25,
            add_residual=False,
        ):
        super().__init__()

        self.use_compressed_info = use_compressed_info
        if use_compressed_info:
            input_dim = 3
        else:
            input_dim = 2

        self.decision_reference_type = decision_reference_type
        if decision_reference_type:
            input_dim += 3

        self.use_norm = decision_mlp_use_norm
        if self.use_norm:
            input_dim += 1

        self.blocks = nn.ModuleList()
        self.blocks_add_residual = []
        if layer_pattern is not None:
            print(f"Priotorizing layer pattern {layer_pattern}")

            dims = []
            last_dim = input_dim
            linear_count = layer_pattern.count('l')
            for i in range(linear_count):
                if i == linear_count - 1:
                    output_dim = decision_mlp_out_dim
                else:
                    output_dim = inner_dim
                dims.append((last_dim, output_dim))
                last_dim = output_dim
            print(f"Layer dims: {dims}")

            last_dim = input_dim
            block = []
            add_residual = False
            for i in range(len(layer_pattern)):
                if layer_pattern[i] == 'l':
                    if block is not None:
                        block = nn.Sequential(*block)
                        self.blocks.append(block)
                        self.blocks_add_residual.append(add_residual)

                    input_dim, output_dim = dims.pop(0)

                    # Create new block
                    block = []
                    add_residual = add_residual and input_dim == output_dim

                    block.append(nn.Linear(input_dim, output_dim, bias=not decision_mlp_disable_bias))
                    last_dim = output_dim
                elif layer_pattern[i] == 'r':
                    block.append(nn.ReLU())
                elif layer_pattern[i] == 'b':
                    block.append(nn.BatchNorm1d(last_dim))
                elif layer_pattern[i] == 'd':
                    block.append(nn.Dropout(dropout))
                elif layer_pattern[i] == 'L':
                    block.append(nn.LayerNorm(last_dim))
                else:
                    raise ValueError(f"Invalid layer pattern {layer_pattern[i]}")
        else:
            raise ValueError("Layer pattern must be provided")

        # Add last block
        block = nn.Sequential(*block)
        self.blocks.append(block)
        self.blocks_add_residual.append(add_residual)
        
        if decision_initialize is not None:
            if decision_initialize == "adafuse":
                def initialize_adafuse(submodule):
                    if isinstance(submodule, torch.nn.Linear):
                        torch.nn.init.normal_(submodule.weight, 0, 0.001)
                        torch.nn.init.constant_(submodule.bias, 0)
                    return
                self.blocks.apply(initialize_adafuse)
        

    def forward(self, importance, similarity, compressed_map, ref_type=None, **kwargs):
        if self.use_norm:
            similarity, norm = similarity

        B, N, N_ = similarity.shape
        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        if self.use_compressed_info:
            mlp_input = torch.cat(
                (
                    importance.view(B, N, 1),
                    most_similar_score.view(B, N, 1),
                    compressed_map.view(B, N, 1)
                ),
                dim=-1
            )
        else:
            mlp_input = torch.cat(
                (
                    importance.view(B, N, 1),
                    most_similar_score.view(B, N, 1)
                ),
                dim=-1
            )

        if ref_type is not None:
            assert self.decision_reference_type, "Reference type is not enabled"
            mlp_input = torch.cat(
                (
                    mlp_input,
                    ref_type.unsqueeze(1).expand(-1, N, -1)
                ),
                dim=-1
            )

        if self.use_norm:
            mlp_input = torch.cat((mlp_input, norm), dim=-1)

        # [B, 2, N-1]
        return self.forward_inner(mlp_input), most_similar_idx, None

    def forward_inner(self, mlp_input):
        B, N, dim = mlp_input.shape
        x = mlp_input.view(B*N, dim)

        for block, add_residual in zip(self.blocks, self.blocks_add_residual):
            if add_residual:
                residual = x
            x = block(x)
            if add_residual:
                x = x + residual

        reuse_decision = x.view(B, N, -1)
        return reuse_decision 

class HeadwiseThreshold(nn.Module):
    def __init__(
            self,
            kept_num,
            num_heads,
            disable_monotonicity=False,
            use_compressed_info=False,
            threshold_act='',
        ):
        super().__init__()
        # self.sim_threshold = nn.Parameter(torch.zeros((kept_num,)), requires_grad=True)
        self.disable_monotonicity = disable_monotonicity
        
        self.sim_threshold = nn.Parameter(torch.empty((num_heads, kept_num)))
        init_threshold(self.sim_threshold, threshold_act, disable_monotonicity)
        self.threshold_act = threshold_act

        self.use_compressed_info = use_compressed_info
        self.linear = nn.Linear(num_heads, 1, bias=False)
        nn.init.uniform_(self.linear.weight, 0, 1 / num_heads)

    def forward(self, importance, similarity, compressed_map, **kwargs):
        # Shape: [H, N]
        threshold = get_monotonic_threshold(
            self.sim_threshold,
            act=self.threshold_act,
            disable_monotonicity=self.disable_monotonicity
        )
        # B, H, N
        importance = torch.argsort(importance, dim=-1, descending=True)
        rank = torch.argsort(importance, dim=-1, descending=False) # [B, N-1]

        # When similarity is lower than threshold, use newly computed value
        B, H, N_1 = rank.shape
        sorted_threshold = torch.gather(
            threshold.unsqueeze(0).expand(B, -1, -1),
            dim=-1,
            index=rank
        )

        if self.use_compressed_info:
            sorted_threshold -= compressed_map
        # [B, H, N-1] => [B, H, N-1, 1]
        sorted_threshold = sorted_threshold.unsqueeze(-1)
        sim_diff = similarity - sorted_threshold
        # [B, H, N-1, N'] => [B, N', N-1, H]
        sim_diff = sim_diff.transpose(1, -1)
        # [B, N', N-1, H] => [B, N', N-1, 1]
        reuse_decision = self.linear(sim_diff)
        reuse_decision = 10*torch.tanh(reuse_decision)
        # [B, N', N-1, 1] => [B, N', N-1] => [B, N-1]
        most_similar_score, most_similar_idx = reuse_decision.squeeze(-1).max(dim=1)

        most_similar_score = most_similar_score.unsqueeze(-1)

        return most_similar_score, most_similar_idx, sorted_threshold


class TopKDecision(nn.Module):
    def __init__(self, k, use_norm=False):
        super().__init__()
        self.k = k
        self.use_norm = use_norm

    def forward(self, importance, similarity, compressed_map, **kwargs):
        if self.use_norm:
            similarity, norm = similarity

        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        topk = torch.topk(most_similar_score, self.k, dim=1)

        B, N, _ = similarity.shape

        reuse_decision = torch.ones_like(most_similar_score)
        reuse_decision = torch.scatter(reuse_decision, 1, topk.indices, 0)
        reuse_decision = reuse_decision.unsqueeze(-1)

        return reuse_decision, most_similar_idx, None

class TldrDecision(nn.Module):
    def __init__(self, k, use_norm=False):
        super().__init__()
        self.k = k
        self.use_norm = use_norm

    def forward(self, importance, similarity, compressed_map, **kwargs):

        similarity = (similarity + 1) / 2 # Move to [0, 1]

        most_similar_score, most_similar_idx = similarity.max(dim=-1)
        edge_idx = most_similar_idx.argsort(dim=-1, descending=True)
        unm_idx = edge_idx[..., self.k:, :] # Recomputed Tokens
        src_idx = edge_idx[..., :self.k, :] # Reused Tokens

        src_so = importance.gather(dim=-1, index=src_idx)

        # Wait, the importance doesn't seem to participate in the decision making?

        return most_similar_score, most_similar_idx, None