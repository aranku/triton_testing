import torch
import triton
import triton.language as tl

from triton_testing.multiquery_attention import _bwd_multiquery_preprocess, _bwd_multiquery_kernel
from triton_testing.query_norm import _querynorm_bwd_dx_fused, _querynorm_bwd_dwdb

@triton.jit
def _fwd_multiquery_querynorm_kernel(
    Q, K, V, sm_scale, n_heads,
    L, M,
    Out,
    weight, bias, eps,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kn, stride_kk,
    stride_vz, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    curr_batch = off_hz // n_heads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = curr_batch * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = curr_batch * stride_vz + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Query norm
    q_unscaled = tl.load(q_ptrs)
    q_mean = tl.sum(q_unscaled, axis=1) / stride_qm
    q_zero_mean = q_unscaled-q_mean
    q_var = tl.sum(q_zero_mean*q_zero_mean, axis=1) / stride_qm
    q_rstd = 1 / tl.sqrt(q_var + eps)
    curr_head = off_hz%n_heads
    w = tl.load(weight + curr_head)
    b = tl.load(bias + curr_head)
    q_hat = (q_unscaled - q_mean) * q_rstd
    q = q_hat * w + b
    q = q.to(tl.float16)
    
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    
class MultiqueryQueryNormAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, weight, bias, eps):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        n_heads = q.shape[1]
        sm_scale = q.shape[-1]**-0.5
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        _fwd_multiquery_querynorm_kernel[grid](
            q, k, v, sm_scale, n_heads,
            L, m,
            o, weight, bias, eps,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk, num_warps=num_warps,
            num_stages=2,
        )
        ctx.save_for_backward(q, k, v, o, L, m, weight, bias)
        ctx.grid = grid
        ctx.eps = eps
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


multiquery_querynorm_flash_attention = MultiqueryQueryNormAttention.apply