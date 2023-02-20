# triton_testing

## Installation

Install triton from source https://triton-lang.org/master/getting-started/installation.html

then run `pip install -e .`

## Tests

Run pytests to check for equality between triton kernels and reference implementation

## Benchmarks

See `benchmark.py` to run kernel benchmarks

### Triton Multiquery Attention vs FlashAttention
#### Forward pass

![alt text](https://github.com/aranku/triton_testing/blob/main/images/multiquery-attention-fwd.png)
#### Backward pass

![alt text](https://github.com/aranku/triton_testing/blob/main/images/multiquery-attention-bwd.png)

### Triton QueryNorm vs Torch QueryNorm
#### Forward pass

![alt text](https://github.com/aranku/triton_testing/blob/main/images/query-norm-forward.png)
#### Backward pass

![alt text](https://github.com/aranku/triton_testing/blob/main/images/query-norm-backward.png)

### Fused QueryNorm + Multiquery Attention vs Unfused QueryNorm + Multiquery Attention

Backward pass remains unfused due to minimal gains from fusing forward pass

#### Forward pass

![alt text](https://github.com/aranku/triton_testing/blob/main/images/fused_multiquery_querynorm-fwd.png)
