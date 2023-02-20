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
    
    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q, k, v, o, l, m, weight, bias = ctx.saved_tensors
        do = do.contiguous()
        dq1 = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        locks = torch.zeros(ctx.grid[0], dtype=torch.int32, device='cuda')
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_multiquery_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do, l,
            do_scaled, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_multiquery_kernel[(ctx.grid[1],ctx.grid[0])](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq1, dk, dv,
            l, m,
            delta, locks,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0], 
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            num_stages=1,
        )
        
        # heuristics for amount of parallel reduction stream for DW/DB
        L = q.shape[2]
        H = weight.shape[0]
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        q_arg = q.reshape(-1, q.shape[-1])
        M, N = q_arg.shape
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // q.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # allocate output
        _dw = torch.empty((H, M//H), dtype=q.dtype, device=weight.device)
        _db = torch.empty((H, M//H), dtype=q.dtype, device=weight.device)
        dq2 = torch.empty_like(dq1)
        _querynorm_bwd_dx_fused[(M,)](dq2, dq1, _dw, _db, q, weight, bias,
                                        q_arg.stride(0), _dw.stride(0), N, L, H, ctx.eps,
                                        BLOCK_SIZE_N=BLOCK_SIZE,
                                        num_warps=num_warps)
        dw = torch.empty((H,), dtype=weight.dtype, device=weight.device)
        db = torch.empty((H,), dtype=weight.dtype, device=weight.device)
        # accumulate partial sums in separate kernel
        _querynorm_bwd_dwdb[(H,)](_dw, _db, dw, db, M, H, _dw.stride(0),
                                    BLOCK_SIZE=32)
        
        return dq2, dk, dv, dw, db, None


multiquery_querynorm_flash_attention = MultiqueryQueryNormAttention.apply