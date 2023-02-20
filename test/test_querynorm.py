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
def test_querynorm_op(BATCH, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    w = torch.randn((1,H,1,1), dtype=dtype, device="cuda", requires_grad=True)
    b = torch.randn((1,H,1,1), dtype=dtype, device="cuda", requires_grad=True)
    dy = .1 * torch.randn_like(q)
    eps = 1e-6
    # naive torch implementation
    naive_out = naive_query_norm(q, w, b, eps)
    # triton implementation
    tri_out = fused_query_norm(q, w.flatten(), b.flatten(), eps)
    tri_out.backward(dy, retain_graph=True)
    dq_tri, dw_tri, db_tri = [_.grad.clone() for _ in [q, w, b]]
    q.grad, w.grad, b.grad = None, None, None
    # backward pass (torch)
    naive_out.backward(dy, retain_graph=True)
    dq_ref, dw_ref, db_ref = [_.grad.clone() for _ in [q, w, b]]
    # compare
    triton.testing.assert_almost_equal(naive_out, tri_out)
    print("Passed all equality checks")