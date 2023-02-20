import pytest
import torch
from torch import einsum, nn
import triton
import triton.language as tl
from triton_testing.multihead_attention import multihead_flash_attention
from triton_testing.multiquery_attention import multiquery_flash_attention
from triton_testing.query_norm import fused_query_norm, naive_query_norm
from triton_testing.fused_multiquery_querynorm import multiquery_querynorm_flash_attention

@pytest.mark.parametrize('BATCH, H, N_CTX, D_HEAD', [(3, 16, 1024, 64)])
def test_multiquery_querynorm_op(BATCH, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    w = torch.randn(H, dtype=dtype, device="cuda", requires_grad=True)
    b = torch.randn(H, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-6
    dout = torch.randn_like(q)
    # triton implementation
    tri_out = multiquery_querynorm_flash_attention(q, k, v, w, b, eps)
    # reference implementation
    ref_out = multiquery_flash_attention(fused_query_norm(q, w, b, eps),k,v)
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)
    print("Passed all equality checks")