import pytest
import torch
from torch import einsum, nn
import triton
import triton.language as tl
from triton_testing.multihead_attention import multihead_flash_attention
from triton_testing.multiquery_attention import multiquery_flash_attention
from triton_testing.query_norm import fused_query_norm, naive_query_norm
from triton_testing.fused_multiquery_querynorm import multiquery_querynorm_flash_attention

@pytest.mark.parametrize('BATCH, H, N_CTX, D_HEAD', [(3, 8, 1024, 64)])
def test_multiquery_op(BATCH, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.randn_like(q)
    # triton implementation
    tri_out = multiquery_flash_attention(q, k, v)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # reference implementation
    reshaped_k = k.reshape(BATCH, N_CTX, 1, D_HEAD).transpose(1, 2).repeat(1,H,1,1) # (B, nh, T, hs)
    reshaped_v = v.reshape(BATCH, N_CTX, 1, D_HEAD).transpose(1, 2).repeat(1,H,1,1) # (B, nh, T, hs)
    ref_out = multihead_flash_attention(q,reshaped_k,reshaped_v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)
    triton.testing.assert_almost_equal(ref_dq, tri_dq)
    triton.testing.assert_almost_equal(ref_dk, tri_dk)
    triton.testing.assert_almost_equal(ref_dv, tri_dv)
    print("Passed all equality checks")