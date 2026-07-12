"""
test_gbs_attention.py
---------------------
Unit tests for GBS attention components.

Run with:
    python -m pytest test_gbs_attention.py -v
or:
    python test_gbs_attention.py
"""

import sys
import torch
import pytest
import numpy as np

from gbs_attention import (
    encode_graph_to_unitary_torch,
    compute_B_matrix,
    GBSAttentionLayer,
)

# ---- Helpers ---------------------------------------------------------------

def rand_sym_adj(N=6, seed=42):
    """Random symmetric adjacency matrix with zero diagonal."""
    torch.manual_seed(seed)
    A = torch.randn(N, N)
    A = (A + A.T) / 2.0
    A.fill_diagonal_(0.0)
    return A

def rand_batch_adj(B=4, N=6, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(B, N, N)
    A = (A + A.transpose(-2,-1)) / 2.0
    eye = torch.eye(N).unsqueeze(0)
    A = A.masked_fill(eye.bool(), 0.0)
    return A

# ---- encode_graph_to_unitary_torch tests -----------------------------------

class TestEncodeGraph:

    def test_T_is_orthogonal_single(self):
        """T @ T^T should be identity (orthogonal matrix)."""
        A = rand_sym_adj(N=6)
        T, r = encode_graph_to_unitary_torch(A)
        I = torch.eye(6)
        err = (T @ T.T - I).abs().max().item()
        assert err < 1e-5, f"T is not orthogonal: max error = {err:.2e}"

    def test_T_is_orthogonal_batched(self):
        """Batched T matrices should each be orthogonal."""
        A = rand_batch_adj(B=8, N=6)
        T, r = encode_graph_to_unitary_torch(A)
        I = torch.eye(6).unsqueeze(0)
        err = (T @ T.transpose(-2,-1) - I).abs().max().item()
        assert err < 1e-5, f"Batched T not orthogonal: max error = {err:.2e}"

    def test_squeezing_nonneg(self):
        """All squeezing parameters r_i must be >= 0."""
        A = rand_sym_adj()
        _, r = encode_graph_to_unitary_torch(A)
        assert (r >= 0).all(), "Negative squeezing values found"

    def test_tanh_r_less_than_one(self):
        """tanh(r_i) must be < 1 (physical constraint for valid GBS state)."""
        A = rand_sym_adj()
        _, r = encode_graph_to_unitary_torch(A)
        assert (r.tanh() < 1.0).all(), "tanh(r) >= 1 violates GBS physical constraint"

    def test_zero_A_gives_zero_squeezing(self):
        """All-zero adjacency matrix should yield zero squeezing (uniform attention)."""
        A = torch.zeros(6, 6)
        _, r = encode_graph_to_unitary_torch(A)
        assert r.abs().max().item() < 1e-6, "Non-zero squeezing for zero adjacency matrix"


# ---- compute_B_matrix tests ------------------------------------------------

class TestBMatrix:

    def test_B_is_symmetric(self):
        """B matrix must be symmetric."""
        A = rand_sym_adj()
        T, r = encode_graph_to_unitary_torch(A)
        B = compute_B_matrix(T, r)
        err = (B - B.T).abs().max().item()
        assert err < 1e-6, f"B is not symmetric: {err:.2e}"

    def test_B_batched_symmetric(self):
        """Batched B matrices must each be symmetric."""
        A = rand_batch_adj(B=4, N=6)
        T, r = encode_graph_to_unitary_torch(A)
        B = compute_B_matrix(T, r)
        err = (B - B.transpose(-2,-1)).abs().max().item()
        assert err < 1e-6, f"Batched B not symmetric: {err:.2e}"


# ---- GBSAttentionLayer tests -----------------------------------------------

class TestGBSAttentionLayer:

    def setup_method(self):
        torch.manual_seed(0)
        self.model = GBSAttentionLayer(nmode=6, d_model=32, c_ratio=0.3)

    def test_output_shape(self):
        """forward(x) should return (B, N, d_model)."""
        x = torch.randn(4, 6, 32)
        out = self.model(x)
        assert out.shape == (4, 6, 32), f"Expected (4, 6, 32), got {out.shape}"

    def test_attn_scores_nonneg(self):
        """GBS attention scores (before softmax) must be non-negative."""
        A = rand_batch_adj(B=2, N=6)
        attn = self.model._get_attn_scores(A)
        assert (attn >= 0).all(), "Negative attention scores found"

    def test_attn_scores_zero_diagonal(self):
        """Attention score diagonal must be zero (no self-loop)."""
        A = rand_batch_adj(B=2, N=6)
        attn = self.model._get_attn_scores(A)
        diag = torch.diagonal(attn, dim1=-2, dim2=-1)
        assert diag.abs().max().item() < 1e-6, "Non-zero diagonal in attention scores"

    def test_attn_scores_symmetric(self):
        """GBS attention score matrix must be symmetric."""
        A = rand_batch_adj(B=2, N=6)
        attn = self.model._get_attn_scores(A)
        err = (attn - attn.transpose(-2,-1)).abs().max().item()
        assert err < 1e-6, f"Attention score matrix not symmetric: {err:.2e}"

    def test_backward_succeeds(self):
        """loss.backward() must complete without error."""
        x = torch.randn(2, 6, 32)
        out = self.model(x)
        loss = out.sum()
        loss.backward()  # Should not raise

    def test_gradient_flows_to_Wq_Wk(self):
        """W_q and W_k must receive non-zero gradients (attention depends on Q, K)."""
        self.model.zero_grad()
        x = torch.randn(2, 6, 32)
        out = self.model(x)
        out.sum().backward()
        assert self.model.W_q.weight.grad is not None, "W_q has no gradient"
        assert self.model.W_k.weight.grad is not None, "W_k has no gradient"
        # Gradients must be non-trivially non-zero
        assert self.model.W_q.weight.grad.abs().sum().item() > 1e-10, "W_q gradient is zero"
        assert self.model.W_k.weight.grad.abs().sum().item() > 1e-10, "W_k gradient is zero"

    def test_gradient_flows_to_Wv(self):
        """W_v must receive non-zero gradient."""
        self.model.zero_grad()
        x = torch.randn(2, 6, 32)
        out = self.model(x)
        out.sum().backward()
        assert self.model.W_v.weight.grad is not None
        assert self.model.W_v.weight.grad.abs().sum().item() > 1e-10

    def test_output_finite(self):
        """Output must not contain NaN or Inf."""
        x = torch.randn(4, 6, 32)
        out = self.model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_zero_input_no_nan(self):
        """All-zero input (pathological case) must not produce NaN."""
        x = torch.zeros(2, 6, 32)
        out = self.model(x)
        assert torch.isfinite(out).all(), "NaN/Inf on zero input"


# ---- Run directly ----------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tests = [
        # (class, method)
        (TestEncodeGraph, 'test_T_is_orthogonal_single'),
        (TestEncodeGraph, 'test_T_is_orthogonal_batched'),
        (TestEncodeGraph, 'test_squeezing_nonneg'),
        (TestEncodeGraph, 'test_tanh_r_less_than_one'),
        (TestEncodeGraph, 'test_zero_A_gives_zero_squeezing'),
        (TestBMatrix,     'test_B_is_symmetric'),
        (TestBMatrix,     'test_B_batched_symmetric'),
        (TestGBSAttentionLayer, 'test_output_shape'),
        (TestGBSAttentionLayer, 'test_attn_scores_nonneg'),
        (TestGBSAttentionLayer, 'test_attn_scores_zero_diagonal'),
        (TestGBSAttentionLayer, 'test_attn_scores_symmetric'),
        (TestGBSAttentionLayer, 'test_backward_succeeds'),
        (TestGBSAttentionLayer, 'test_gradient_flows_to_Wq_Wk'),
        (TestGBSAttentionLayer, 'test_gradient_flows_to_Wv'),
        (TestGBSAttentionLayer, 'test_output_finite'),
        (TestGBSAttentionLayer, 'test_zero_input_no_nan'),
    ]

    passed = 0
    failed = 0
    for cls, method in tests:
        obj = cls()
        if hasattr(obj, 'setup_method'):
            obj.setup_method()
        try:
            getattr(obj, method)()
            print(f'  PASS  {cls.__name__}.{method}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {cls.__name__}.{method}')
            traceback.print_exc()
            failed += 1

    print(f'\n{passed} passed, {failed} failed out of {passed+failed} tests.')
    sys.exit(0 if failed == 0 else 1)
