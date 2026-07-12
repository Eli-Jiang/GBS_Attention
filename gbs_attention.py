"""
gbs_attention.py
----------------
Gaussian Boson Sampling (GBS) Attention Layer.

Training path: fully differentiable via B-matrix analytical formula.
    A (QK^T symmetrized) → encode_graph_to_unitary_torch → B = T·diag(tanh r)·T^T
    attn[i,j] = B[i,j]^2  (proportional to 2-photon coincidence probability)

Analysis path: deepquantum MCMC sampling for visualization (no gradient, not used in training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core encoding: A → (T, r)
# ---------------------------------------------------------------------------

def encode_graph_to_unitary_torch(A: torch.Tensor, c_ratio: float = 0.3):
    """
    Differentiable version of encode_graph_to_unitary from precode.ipynb.

    Converts symmetric adjacency matrix A (with zero diagonal) → GBS params.
    Supports both single (N, N) and batched (B, N, N) inputs.

    Args:
        A      : (..., N, N) real symmetric, zero diagonal.
        c_ratio: Squeezing strength scale factor in (0, 1). Default 0.3.

    Returns:
        T: (..., N, N) real orthogonal interferometer matrix.
        r: (..., N)   non-negative squeezing coefficients.
    """
    W = A @ A.transpose(-2, -1)                  # A^2, PSD
    d, U = torch.linalg.eigh(W)                  # d≥0, U real orthogonal
    d = d.clamp(min=0.0)

    # Sign-based phase correction (differentiable alternative to torch.angle)
    # Ensures B ≈ c · |A|_spectral  (all eigenvalue contributions positive)
    inner = U.transpose(-2, -1) @ A @ U          # (..., N, N) real
    B_diag = torch.einsum('...ii->...i', inner)  # (..., N) diagonal elements
    signs = torch.sign(B_diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    # T = U · diag(signs)  — each eigenvector column scaled by ±1
    T = U * signs.unsqueeze(-2)                  # (..., N, N)

    # Squeezing: tanh(r_i) = c · sqrt(d_i),  c chosen so max tanh(r) = c_ratio
    sqrt_d = d.sqrt()
    max_sqrt_d = sqrt_d.amax(dim=-1, keepdim=True).clamp(min=1e-9)
    c = c_ratio / max_sqrt_d                     # (..., 1)
    tanh_r = (c * sqrt_d).clamp(0.0, 0.999999)  # (..., N)
    r = tanh_r.arctanh()                         # (..., N)

    return T, r


def compute_B_matrix(T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    GBS B-matrix: B = T · diag(tanh r) · T^T.

    B[i,j]^2 is the leading-order proxy for the 2-photon coincidence
    probability P(1_i, 1_j) in Gaussian Boson Sampling.

    Args:
        T: (..., N, N) real orthogonal.
        r: (..., N)   squeezing coefficients.

    Returns:
        B: (..., N, N) real symmetric.
    """
    return T @ torch.diag_embed(r.tanh()) @ T.transpose(-2, -1)


# ---------------------------------------------------------------------------
# GBS Attention Layer
# ---------------------------------------------------------------------------

class GBSAttentionLayer(nn.Module):
    """
    Replaces softmax self-attention with a GBS-physics-inspired mechanism.

    Forward path (training, differentiable):
        Q, K, V = x · W_q/k/v
        S = Q K^T / sqrt(d_k)
        A = (S + S^T) / 2,  diag = 0          # adjacency matrix
        T, r = encode_graph_to_unitary_torch(A)
        B = T · diag(tanh r) · T^T
        attn[i,j] = B[i,j]^2                  # ≈ 2-photon coincidence prob
        out = softmax(attn) · V

    Analysis path (no gradient):
        visualize_gbs_sampling(A) → deepquantum MCMC result dict

    Args:
        nmode  : Number of GBS modes = number of patches (N).
        d_model: Feature dimension.
        c_ratio: GBS squeezing strength ∈ (0, 1). Default 0.3 (matches precode).
    """

    def __init__(self, nmode: int = 6, d_model: int = 32, c_ratio: float = 0.3):
        super().__init__()
        self.nmode = nmode
        self.d_model = d_model
        self.c_ratio = c_ratio
        self.scale = d_model ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _get_attn_scores(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute GBS attention matrix from batched adjacency matrices.

        Args:
            A: (B, N, N) symmetric, zero diagonal.

        Returns:
            attn: (B, N, N) non-negative, symmetric, zero diagonal.
        """
        T, r = encode_graph_to_unitary_torch(A, self.c_ratio)
        B = compute_B_matrix(T, r)                          # (B, N, N)
        attn = B.pow(2)                                     # non-negative
        # Zero diagonal
        eye_mask = torch.eye(self.nmode, dtype=torch.bool, device=A.device)
        attn = attn.masked_fill(eye_mask.unsqueeze(0), 0.0)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            out: (B, N, d_model)
        """
        B, N, _ = x.shape
        Q = self.W_q(x)                                     # (B, N, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Symmetric adjacency matrix with zero diagonal
        S = (Q @ K.transpose(-2, -1)) * self.scale         # (B, N, N)
        A = (S + S.transpose(-2, -1)) / 2.0
        eye_mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        A = A.masked_fill(eye_mask, 0.0)

        # GBS attention scores (fully batched, no Python loop)
        attn = self._get_attn_scores(A)                     # (B, N, N)
        attn = F.softmax(attn, dim=-1)

        out = attn @ V                                      # (B, N, d_model)
        return self.out_proj(out)

    @torch.no_grad()
    def visualize_gbs_sampling(self, A: torch.Tensor, shots: int = 2048):
        """
        Run deepquantum MCMC sampling for subgraph visualization.
        NOT used in training.

        Args:
            A    : (N, N) adjacency matrix (CPU, 2D tensor).
            shots: Number of MCMC samples.

        Returns:
            result: {FockState: count} dict from deepquantum.
        """
        import deepquantum as dq
        T, r = encode_graph_to_unitary_torch(A.cpu(), self.c_ratio)
        gbs = dq.photonic.GaussianBosonSampling(
            nmode=self.nmode,
            squeezing=r,
            unitary=T.to(torch.cfloat)
        )
        gbs()
        gbs.detector = 'pnrd'
        return gbs.measure(shots=shots, mcmc=True)
