import torch
from torch import nn
from ...utils.train import normalize_vector

class CosineSimilarity(nn.Module):
    def __init__(self, exp=1, return_norm=False):
        super().__init__()
        self.exp = exp
        self.return_norm = return_norm

    def forward(self, lhs, rhs):
        # Find cosine similarity between query and cached input
        norm_recomputed_input = normalize_vector(lhs, return_norm=self.return_norm)
        if self.return_norm:
            norm_recomputed_input, lhs_norm = norm_recomputed_input
        else:
            lhs_norm = None
        norm_cached_input = normalize_vector(rhs)

        similarity = norm_recomputed_input @ norm_cached_input.transpose(-1, -2)

        similarity = similarity ** self.exp

        return similarity, lhs_norm

    
class HeadwiseCosineSimilarity(nn.Module):
    def __init__(self, num_heads, exp=1):
        super().__init__()
        self.num_heads = num_heads
        self.exp = exp

    def forward(self, lhs, rhs):
        # [B, N, dim] => [B, N, H, head_dim] => [B, H, N, head_dim] => [B*H, N, head_dim]
        B, N, dim = lhs.shape
        H = self.num_heads
        lhs = lhs.view(B, N, H, -1).transpose(1, 2).reshape(B * H, N, -1)
        N_ = rhs.shape[1]
        rhs = rhs.view(B, N_, H, -1).transpose(1, 2).reshape(B * H, N_, -1)

        norm_recomputed_input = normalize_vector(lhs)
        norm_cached_input = normalize_vector(rhs)

        # [B*H, N, head_dim] @ [B*H, head_dim, N_] => [B*H, N, N_]
        similarity = norm_recomputed_input @ norm_cached_input.transpose(-1, -2)
        similarity = similarity ** self.exp

        similarity = similarity.view(B, self.num_heads, N, N_)

        return similarity, None


class L2Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        # Find L2 similarity between query and cached input
        diff = lhs.unsqueeze(2) - rhs.unsqueeze(1)
        similarity = -torch.norm(diff, dim=-1, p=2)

        return similarity, None

class SADSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        # Find L1 similarity between query and cached input
        diff = lhs.unsqueeze(2) - rhs.unsqueeze(1)
        similarity = -torch.abs(diff).sum(dim=-1)

        return similarity, None

class LocalL2Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        B, N, dim = lhs.shape
        _, N_, _ = rhs.shape
        diff = lhs - rhs[:, -N:]
        similarity = torch.norm(diff, dim=-1, p=2)

        ret = torch.full((B, N, N_), -1e9, device=lhs.device)
        ret[:, :, -N:].diagonal(dim1=1, dim2=2)[:] = similarity

        return ret

class LocalCosineSimilarity(nn.Module):
    def __init__(self, exp=1, return_norm=False):
        super().__init__()
        self.exp = exp
        self.return_norm = return_norm

    def forward(self, lhs, rhs):
        B, N_1, dim = lhs.shape
        N = N_1 + 1
        _, N_, _ = rhs.shape

        # Find cosine similarity between query and cached input
        norm_recomputed_input = normalize_vector(lhs, return_norm=self.return_norm)
        if self.return_norm:
            norm_recomputed_input, lhs_norm = norm_recomputed_input
        else:
            lhs_norm = None
        norm_cached_input = normalize_vector(rhs)

        # [B, N-1, dim] @ [B, N_, dim] => [B, N-1, N_]
        similarity = norm_recomputed_input @ norm_cached_input.transpose(-1, -2)
        similarity = similarity ** self.exp

        # [B, N-1, N_] => [B, N-1, F, N] => [B, N-1, F, N]
        F = N_ // N
        mask = torch.eye(N_1, dtype=torch.bool, device=lhs.device).view(1, N_1, 1, N_1).expand(B, -1, F, -1)
        similarity.view(B, N_1, -1, N)[..., -N_1:][~mask] = -1e9 # Tokens should not refer to other tokens
        similarity.view(B, N_1, -1, N)[..., 0] = -1e9              # No one should refer to CLS

        return similarity, lhs_norm