import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_querynorm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    L, # sequence length
    H, # num heads
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr # feature size to next power of 2
    ):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    curr_head = row // L % H

    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + curr_head)
    b = tl.load(B + curr_head)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _querynorm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,   # pointer to the input
    W,   # pointer to the weights
    B,   # pointer to the biases
    Mean,   # pointer to the mean
    Rstd,   # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row of x
    dw_stride, # how much to increase the pointer when moving by 1 row of dw
    N,  # number of columns in X
    L, # sequence length
    H, # number of heads
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_N: tl.constexpr
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = cols_n < N

    curr_head = row // L % H

    X += row * x_stride
    DY += row * x_stride
    DX += row * x_stride
    
    DW += curr_head*dw_stride + row%dw_stride
    DB += curr_head*dw_stride + row%dw_stride
    
    # Load data to SRAM
    x = tl.load(X + cols_n, mask=mask_n, other=0).to(tl.float32)
    dy = tl.load(DY + cols_n, mask=mask_n, other=0).to(tl.float32)
    w = tl.load(W + curr_head).to(tl.float32)
    mean = tl.load(Mean + row) # can recompute
    rstd = tl.load(Rstd + row) # can recompute
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask_n, xhat, 0.)
    wdy = tl.where(mask_n, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols_n, dx, mask=mask_n)
    
    # Accumulate partial sums for dw/db
    partial_dw = tl.sum((dy * xhat).to(w.dtype), axis=0)
    partial_db = tl.sum((dy).to(w.dtype), axis=0)
    tl.store(DW, partial_dw)
    tl.store(DB, partial_db)

@triton.jit
def _querynorm_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    H,  # number of heads
    dw_stride, # how much to increase the pointer when moving by 1 row of dw
    BLOCK_SIZE: tl.constexpr
):
    # Map the program id to the elements of DW and DB it should compute.
    curr_head = tl.program_id(0)
    num_cols = M//H
    # Compute mean
    _dw_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _db_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, num_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        dw = tl.load(DW + curr_head*dw_stride + cols, mask=cols < num_cols, other=0.).to(tl.float32)
        db = tl.load(DB + curr_head*dw_stride + cols, mask=cols < num_cols, other=0.).to(tl.float32)
        _dw_sum += dw
        _db_sum += db
    dw_sum = tl.sum(_dw_sum, axis=0)
    db_sum = tl.sum(_db_sum, axis=0)
    tl.store(FINAL_DW + curr_head, dw_sum)
    tl.store(FINAL_DB + curr_head, db_sum)

class FusedQueryNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        B, H, L, N = x.shape
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, N)
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _fwd_querynorm_kernel[(M,)](x_arg, y, weight, bias, mean, rstd,
                                  x_arg.stride(0), N, L, H, eps,
                                  BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        L = x.shape[2]
        H = weight.shape[0]
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # allocate output
        _dw = torch.empty((H, M//H), dtype=x.dtype, device=weight.device)
        _db = torch.empty((H, M//H), dtype=x.dtype, device=weight.device)
        dx = torch.empty_like(dy)
        _querynorm_bwd_dx_fused[(M,)](dx, dy, _dw, _db, x, weight, bias, mean, rstd,
                                        x_arg.stride(0), _dw.stride(0), N, L, H, ctx.eps,
                                        BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                        num_warps=ctx.num_warps)
        dw = torch.empty((H,), dtype=weight.dtype, device=weight.device)
        db = torch.empty((H,), dtype=weight.dtype, device=weight.device)
        # accumulate partial sums in separate kernel
        _querynorm_bwd_dwdb[(H,)](_dw, _db, dw, db, M, H, _dw.stride(0),
                                    BLOCK_SIZE=32)
        return dx, dw, db, None


fused_query_norm = FusedQueryNorm.apply

def naive_query_norm(x, weight, bias, eps):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True, unbiased=False)
    return weight*(x - mean)/(std + eps) + bias