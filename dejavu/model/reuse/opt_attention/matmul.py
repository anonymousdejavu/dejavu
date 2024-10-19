import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size', 'gm', 'gk'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    hidden_states_norm_ptr,
    reference_states_norm_ptr,
    similarity_ptr,
    reference_cache_len_ptr,
    # batch strides
    hidden_states_norm_batch_stride,
    reference_states_norm_batch_stride,
    similarity_batch_stride,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    lda, ldb, ldc,
    group_size,
    gm, gk,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)

    for g in range(group_size):
        # get the gemm size of the current problem
        reference_cache_len = tl.load(reference_cache_len_ptr + g).to(tl.int32)
        num_n_tiles = tl.cdiv(reference_cache_len, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            a_ptr = hidden_states_norm_ptr + g * hidden_states_norm_batch_stride
            b_ptr = reference_states_norm_ptr + g * reference_states_norm_batch_stride
            c_ptr = similarity_ptr + g * similarity_batch_stride
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            mask_am = offs_am < gm
            mask_bn = offs_bn < reference_cache_len

            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_bn[:, None] * ldb + offs_k[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # tl.multiple_of(a_ptrs, [32, 32])
                # tl.multiple_of(b_ptrs, [32, 32])
                mask_k = offs_k < gk - kk * BLOCK_SIZE_K
                mask_a = mask_am[:, None] & mask_k[None, :]
                mask_b = mask_bn[:, None] & mask_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.)
                b = tl.load(b_ptrs, mask=mask_b, other=0.)
                accumulator += tl.dot(a, b.T)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            c_mask = (offs_am[:, None] < gm) & (offs_bn[None, :] < reference_cache_len)
            tl.store(c_ptrs, accumulator, mask=c_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(hidden_states_norm, reference_states_norm, reference_cache_len):
    device = torch.device('cuda')

    B, N, dim = hidden_states_norm.shape
    assert reference_states_norm.shape == (B, 5*N, dim)
    assert reference_cache_len.shape == (B,)

    similarity = torch.zeros((B, N, 5*N), device=device, dtype=hidden_states_norm.dtype)

    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        hidden_states_norm,
        reference_states_norm,
        similarity,
        reference_cache_len,
        hidden_states_norm_batch_stride=hidden_states_norm.stride(0),
        reference_states_norm_batch_stride=reference_states_norm.stride(0),
        similarity_batch_stride=similarity.stride(0),
        lda=hidden_states_norm.stride(1),
        ldb=reference_states_norm.stride(1),
        ldc=similarity.stride(1),
        group_size=B,
        gm=N,
        gk=dim
    )

    return similarity

if __name__ == '__main__':
    B, N, dim = 64, 128, 256
    hidden_states_norm = torch.randn((B, N, dim), device='cuda', dtype=torch.float32)
    # hidden_states_norm = torch.arange(B*N*dim, device='cuda', dtype=torch.float32).reshape(B, N, dim)
    reference_states_norm = torch.randn((B, 5*N, dim), device='cuda', dtype=torch.float32)
    # reference_states_norm = torch.arange(B*dim*5*N, device='cuda', dtype=torch.float32).reshape(B, 5*N, dim)
    reference_cache_len = torch.randint(1, 5*N, (B,), device='cuda')

    # similarity = group_gemm_fn(hidden_states_norm, reference_states_norm, reference_cache_len)
    # print(similarity.shape)

    reference_cache_len[:] = 5*N
    similarity = group_gemm_fn(hidden_states_norm, reference_states_norm, reference_cache_len)
    torch_similarity = torch.bmm(hidden_states_norm, reference_states_norm.transpose(1, 2))
    diff = similarity - torch_similarity

    print(diff.abs().max().item())