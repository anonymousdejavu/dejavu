import torch

import triton
import triton.language as tl


def self_attn_per_frame(query_states, key_states, value_states, gather_indices, num_heads):
    H, N_, head_dim = query_states.shape
    F, B, N = gather_indices.shape

    qk = torch.bmm(query_states, key_states.transpose(1, 2))
    qk_max = qk.max(dim=-1, keepdim=True).values
    qk = qk - qk_max
    qk_exp = qk.exp()

    # qk_exp: [H, N', N'] => [1, H, N', N'] => [1, 1, H, N', N'] => [4, B, H, N', N']
    # gather_indices: [4, B, 197] => [4, B, 1, 1, 197] => [4, B, num_heads, N', 197]
    # qk_gathered: [4, B, H, N', 197]
    qk_gathered = torch.gather(
        qk_exp.unsqueeze(0).unsqueeze(0).expand(F, B, -1, -1, -1),
        dim=4,
        index=gather_indices.unsqueeze(2).unsqueeze(2).expand(-1, -1, num_heads, N_, -1)
    )
    # qk_gathered: [4, B, H, N', 197]
    # gather_indices: [4, B, 197] => [4, B, 1, 197, 1] => [4, B, H, 197, 197]
    qk_gathered = torch.gather(
        qk_gathered,
        dim=3,
        index=gather_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, num_heads, -1, N)
    )
    # qk_gathered: [4, B, H, 197, 197] => [4, B, H, 197, 1]
    qk_gathered_sum = qk_gathered.sum(dim=-1, keepdim=True)
    attn_weights = qk_gathered / qk_gathered_sum

    # attn_weights: [4, B, H, N, N] => [4*B*H, N, N]
    # value_states: [4*B, H, N, head_dim] =>  [4*B*H, N, head_dim]
    # attn_output: [4*B*H, N, head_dim]
    attn_output = torch.bmm(attn_weights.view(-1, N, N), value_states.view(-1, N, head_dim))
    # [4*B*H, N, head_dim] => [4*B, H, N, head_dim]
    attn_output = attn_output.view(-1, num_heads, N, head_dim)

    return attn_weights, attn_output


def self_attn_restore_first(query_states, key_states, value_states, gather_indices, num_heads, eps=1e-6):
    BH, N_, head_dim = query_states.shape
    F, B, N = gather_indices.shape

    # query_states: [H, N', head_dim] => [1, 1, H, N', head_dim] => [4, B, H, N', head_dim]
    # gater_indices: [4, B, N] => [4, B, 1, N, 1] => [4, B, H, N, head_dim]
    query_states = torch.gather(
        query_states.unsqueeze(0).unsqueeze(0).expand(F, B, -1, -1, -1),
        dim=3,
        index=gather_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, num_heads, -1, head_dim)
    )
    key_states = torch.gather(
        key_states.unsqueeze(0).unsqueeze(0).expand(F, B, -1, -1, -1),
        dim=3,
        index=gather_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, num_heads, -1, head_dim)
    )

    qk = torch.bmm(
        query_states.view(-1, N, head_dim),
        key_states.view(-1, N, head_dim).transpose(1, 2)
    )

    # attn_weights: [4*B*H, N, N]
    attn_weights = torch.nn.functional.softmax(qk, dim=-1)

    attn_output = torch.bmm(attn_weights.view(-1, N, N), value_states.view(-1, N, head_dim))
    # [4*B*H, N, head_dim] => [4*B, H, N, head_dim]
    attn_output = attn_output.view(-1, num_heads, N, head_dim)

    # Match the shape of self_attn_per_frame
    attn_weights = attn_weights.view(F, B, num_heads, N, N)
    attn_output = attn_output.view(-1, num_heads, N, head_dim)

    return attn_weights, attn_output

if __name__ == '__main__':
    F=4
    B=2
    N=197
    H=12
    DIM=768
    head_dim = DIM // H

    torch.manual_seed(0)

    # Make plausible gather_idx
    REUSE_RATE = 0.9
    gather_idx = torch.zeros((F, B, N), dtype=torch.long, device='cuda')
    # The gather idxs are mostly from cached frames
    gather_idx[:, :] = torch.arange(B*N).view(1, B, N)

    num_compute = int((1 - REUSE_RATE) * N)
    # Assign new values to some random indices
    N_ = B*N
    for i in range(4):
        for j in range(B):
            for k in range(num_compute):
                idx = torch.randint(0, N, (1,)).item()
                gather_idx[i, j, idx] = N_
                N_ += 1 # New compute states

    query_states = torch.randn((H, N_, head_dim), dtype=torch.float32, device='cuda')
    key_states = torch.randn((H, N_, head_dim), dtype=torch.float32, device='cuda')
    value_states = torch.randn((F*B, H, N, head_dim), dtype=torch.float32, device='cuda')

    restore_last = self_attn_per_frame(query_states, key_states, value_states, gather_idx, H)
    restore_first = self_attn_restore_first(query_states, key_states, value_states, gather_idx, H)

    restore_last_weights = restore_last[0].view(F, B, H, N, N)
    restore_first_weights = restore_first[0].view(F, B, H, N, N)

    diff_weights = restore_last_weights - restore_first_weights

    print(restore_last_weights.max())
    print(torch.allclose(restore_last_weights, restore_first_weights, atol=1e-7))

    restore_last_output = restore_last[1].view(F, B, H, N, head_dim)
    restore_first_output = restore_first[1].view(F, B, H, N, head_dim)

    print(restore_last_output.max())
    print(torch.allclose(restore_last_output, restore_first_output, atol=1e-6))



    













