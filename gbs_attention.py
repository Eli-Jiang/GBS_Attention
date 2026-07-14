"""
gbs_attention.py — GBS（高斯玻色采样）量子注意力层
=====================================================

训练路径（全可微）：
    A（QK^T 对称化）→ encode_graph_to_unitary_torch → B = T·diag(tanh r)·T^T
    attn[i,j] = B[i,j]²  ← 2-光子共现概率的解析近似，可通过 autograd 反传

分析路径（不计算梯度）：
    visualize_gbs_sampling()  ← 调用 deepquantum MCMC 采样，用于可视化子图检测

物理意义：
    c_ratio 控制 GBS 挤压强度，决定每个模式的平均光子数。
    B 矩阵是 GBS 态的邻接矩阵表示，B[i,j]² ∝ 模式 i 和 j 上同时探测到单光子的概率。
    低 c_ratio → B 接近线性变换（tanh r ≈ r）；高 c_ratio → 更丰富的多体关联但数值范围更大。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 全局 GBS 物理参数配置
# ============================================================
GBS_CFG = {
    # === c_ratio 模式开关 ===
    # "fixed"     — c_ratio 字段的值作为固定超参数（默认，保持原行为）
    # "learnable" — c_ratio 作为初始值，模型自动学习最优挤压强度
    "c_ratio_mode": "fixed",          # "fixed" | "learnable"

    # GBS 挤压强度缩放因子：tanh(r_i) = c_ratio / max_sqrt_d * sqrt(d_i)
    # 取值范围 (0, 1)；越小越接近线性注意力，越大非线性越强
    # fixed 模式下直接使用；learnable 模式下作为初始值
    "c_ratio": 0.3,

    # tanh(r_i) 的数值裁剪上界：防止 arctanh 发散
    "tanh_r_max": 0.999,

    # W = A @ A^T 特征值的数值下界：防止 sqrt(0) 和除零
    "eig_min_clamp": 1e-12,

    # eigh 正则化强度：W = A² 在 A 零迹（对角线=0）时必然存在结构性退化特征值
    # （λᵢ = -λⱼ 的情况导致 λᵢ² = λⱼ²），加 ε·I 可打破退化，稳定 CUDA cuSOLVER
    # N 越大（patch 数越多），退化越容易触发，因此此参数不可省去
    "eig_regularize": 1e-5,

    # max_sqrt_d 的数值下界：防止零 A 矩阵导致除零
    "max_sqrt_d_min": 1e-9,

    # 可视化分析时的默认 MCMC 采样数
    "default_shots": 2048,
}


# ============================================================
# 核心函数：邻接矩阵 A → GBS 参数 (T, r)
# ============================================================

def encode_graph_to_unitary_torch(
    A: torch.Tensor,
    c_ratio: "float | torch.Tensor" = GBS_CFG["c_ratio"]
):
    """
    将对称邻接矩阵 A 编码为 GBS 参数（T 和 r）。
    与 precode.ipynb 中的 encode_graph_to_unitary 等价，但全程使用 PyTorch 算子，
    支持自动微分（梯度可通过 torch.linalg.eigh 反传）。

    支持批量输入（batch 维度在最前面）。

    物理步骤：
        1. W = A² = A @ A^T（正半定，特征值 ≥ 0）
        2. 对 W 做谱分解：W = U diag(d) U^T
        3. 符号校正：让 B 矩阵近似 c·|A|（谱绝对值形式）
        4. 挤压系数：tanh(r_i) = c · sqrt(d_i)，c 使最大 tanh(r) = c_ratio

    Args:
        A       : (..., N, N) 实对称张量，对角线为 0。
        c_ratio : GBS 挤压强度缩放因子，取 (0, 1)。支持 float 或 Tensor（用于可学习模式）。

    Returns:
        T : (..., N, N) 实正交矩阵（干涉仪酉矩阵的实数形式）。
        r : (..., N)   非负挤压系数。
    """
    # Step 1: 计算 W = A^2（正半定）
    W = A @ A.transpose(-2, -1)

    # Step 2: 谱分解
    # 注意：由于 A 的对角线为 0 → trace(A) = 0 → A 的特征值之和为 0
    # 这意味着若 λᵢ ≈ -λⱼ，则 W = A² 的 λᵢ² ≈ λⱼ²，出现近似退化特征值
    # 对 N=12 等较大矩阵，退化概率显著提升，会导致 CUDA cuSOLVER 在连续调用后不稳定
    # 修正方案：先强制对称化（消除浮点误差引入的微小非对称），再加 ε·I 打破退化
    W = (W + W.transpose(-2, -1)) * 0.5                          # 强制对称

    # ε·I 正则化：用 max|A| 缩放，确保 A=0 时无正则化（保持物理正确性：无图→无挤压）
    # A≠0 时正则化与 A 的量级匹配，有效打破近退化而不影响主要特征值
    _scale = A.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=0.0)  # (..., 1, 1)
    N = W.shape[-1]
    W = W + GBS_CFG["eig_regularize"] * _scale * torch.eye(
        N, device=W.device, dtype=W.dtype
    )
    
    # 强力打破特征值严格退化：cuSOLVER 对完全相等的特征值（如 W 全 0）极其敏感
    # 加入一个极小的非均匀对角线噪声矩阵，使得所有特征值在底层都不完全相等
    diag_noise = torch.arange(1, N + 1, device=W.device, dtype=W.dtype) * 1e-6
    W = W + torch.diag(diag_noise)

    try:
        d, U = torch.linalg.eigh(W)          # d: (..., N), U: (..., N, N)
    except torch._C._LinAlgError:
        # CUDA 上的 cuSOLVER 若依然崩溃，则 fallback 到 float64 双精度求解
        try:
            d, U = torch.linalg.eigh(W.to(torch.float64))
            d, U = d.to(W.dtype), U.to(W.dtype)
        except Exception:
            # 终极 Fallback：到 CPU 上使用更稳定的 LAPACK 求解
            W_cpu = W.cpu().to(torch.float64)
            d_cpu, U_cpu = torch.linalg.eigh(W_cpu)
            d, U = d_cpu.to(W.device, W.dtype), U_cpu.to(W.device, W.dtype)
            
    d = d.clamp(min=GBS_CFG["eig_min_clamp"])

    # Step 3: 符号校正（可微替代 torch.angle）
    # B_diag[..., i] = [U^T A U]_ii 表示第 i 个特征方向上 A 的贡献符号
    # 将各特征向量列乘以对应符号，使 B 矩阵近似 c · |A|_spectral
    inner = U.transpose(-2, -1) @ A @ U              # (..., N, N)
    B_diag = torch.einsum('...ii->...i', inner)       # (..., N) 对角元素
    signs = torch.sign(B_diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    T = U * signs.unsqueeze(-2)                       # (..., N, N) 列乘符号

    # Step 4: 计算挤压系数 r
    sqrt_d = d.sqrt()
    max_sqrt_d = sqrt_d.amax(dim=-1, keepdim=True).clamp(
        min=GBS_CFG["max_sqrt_d_min"]
    )
    c = c_ratio / max_sqrt_d                          # (..., 1)
    tanh_r = (c * sqrt_d).clamp(0.0, GBS_CFG["tanh_r_max"])
    r = tanh_r.arctanh()

    return T, r


def compute_B_matrix(T: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    计算 GBS B 矩阵：B = T · diag(tanh r) · T^T。

    物理意义：
        B[i,j]² ≈ P(1_i, 1_j)（2-光子共现概率的前导阶近似）
        即模式 i 和模式 j 同时探测到单光子的概率与 |B[i,j]|² 成正比。

    Args:
        T : (..., N, N) 实正交矩阵。
        r : (..., N)   挤压系数（由 encode_graph_to_unitary_torch 得到）。

    Returns:
        B : (..., N, N) 实对称矩阵。
    """
    return T @ torch.diag_embed(r.tanh()) @ T.transpose(-2, -1)


# ============================================================
# GBS 注意力层
# ============================================================

class GBSAttentionLayer(nn.Module):
    """
    GBS（高斯玻色采样）注意力层 — 可替换标准 Softmax 自注意力。

    正向传播（训练，全可微）：
        Q, K, V = x · W_q / W_k / W_v
        S = Q K^T / sqrt(d_k)              相似度矩阵
        A = (S + S^T) / 2,  diag = 0       对称邻接矩阵
        T, r = encode_graph_to_unitary_torch(A)
        B = T · diag(tanh r) · T^T         GBS B 矩阵
        attn[i,j] = B[i,j]²               2-光子概率代理（非负、对称）
        out = softmax(attn) · V

    可扩展性说明：
        - nmode 自动跟随输入 N（由 GBSPatchTransformer 初始化时传入 num_patches）
        - c_ratio 可在模型层面调整（不影响其他组件）
        - visualize_gbs_sampling() 提供 deepquantum MCMC 分析路径，与训练完全解耦

    Args:
        nmode     : GBS 模式数 = patch 数量 N，支持任意正整数。
        d_model   : 特征维度。
        c_ratio   : GBS 挤压强度取 (0, 1)。
        c_ratio_init : learnable 模式的初始值（None 则使用 c_ratio）。
        c_ratio_mode : "fixed" 或 "learnable"（None 则使用 GBS_CFG 默认值）。
    """

    def __init__(
        self,
        nmode: int = 6,
        d_model: int = 32,
        c_ratio: float = GBS_CFG["c_ratio"],
        c_ratio_init: float | None = None,
        c_ratio_mode: str | None = None,
    ):
        super().__init__()
        self.nmode = nmode
        self.d_model = d_model
        self.scale = d_model ** -0.5

        # c_ratio 模式：实例级 > 全局配置级
        self._c_ratio_mode = (
            c_ratio_mode if c_ratio_mode is not None
            else GBS_CFG.get("c_ratio_mode", "fixed")
        )
        _init_val = c_ratio_init if c_ratio_init is not None else c_ratio

        if self._c_ratio_mode == "learnable":
            # sigmoid 逆变换：从 (0,1) 映射到 R，保证梯度自由流动
            _eps = 1e-8
            _logit = math.log(_init_val / (1.0 - _init_val + _eps))
            self._c_logit = nn.Parameter(torch.tensor(_logit, dtype=torch.float32))
        else:
            self._c_logit = None
            self._c_fixed = c_ratio

        # QKV 投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ---- c_ratio 属性：fixed 返回浮点数，learnable 返回 sigmoid(logit) ----
    @property
    def c_ratio(self) -> "torch.Tensor | float":
        if self._c_ratio_mode == "learnable":
            return torch.sigmoid(self._c_logit)  # type: ignore[arg-type]
        return self._c_fixed

    def get_c_ratio_info(self) -> str:
        if self._c_ratio_mode == "learnable":
            cr: float = torch.sigmoid(self._c_logit).item()  # type: ignore[arg-type]
            return f"c_ratio(learnable)={cr:.4f}"
        return f"c_ratio(fixed)={self._c_fixed}"

    def _get_attn_scores(self, A: torch.Tensor) -> torch.Tensor:
        """
        从批量邻接矩阵 A 计算 GBS 注意力分数矩阵。

        Args:
            A : (B, N, N) 实对称，对角线为 0。

        Returns:
            attn : (B, N, N) 非负、对称、对角线为 0 的注意力分数。
        """
        T, r = encode_graph_to_unitary_torch(A, self.c_ratio)   # 编码为 GBS 参数
        B_mat = compute_B_matrix(T, r)                           # GBS B 矩阵
        attn = B_mat.pow(2)                                      # 2-光子概率代理

        # 清零对角线（不允许 token 关注自身在 GBS 图中）
        N = A.shape[-1]
        eye_mask = torch.eye(N, dtype=torch.bool, device=A.device).unsqueeze(0)
        attn = attn.masked_fill(eye_mask, 0.0)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, d_model)  — N = num_patches = nmode

        Returns:
            out : (B, N, d_model)
        """
        B, N, _ = x.shape

        # Step 1: QKV 投影
        Q = self.W_q(x)   # (B, N, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: 计算相似度矩阵并对称化为邻接矩阵
        S = (Q @ K.transpose(-2, -1)) * self.scale    # (B, N, N)
        A = (S + S.transpose(-2, -1)) / 2.0           # 对称化
        eye_mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        A = A.masked_fill(eye_mask, 0.0)              # 清零对角线

        # Step 3: GBS 注意力分数（全批量，无 Python 循环）
        attn = self._get_attn_scores(A)               # (B, N, N)

        # Step 4: Softmax 归一化 + 加权求和 V
        attn = F.softmax(attn, dim=-1)
        out = attn @ V                                # (B, N, d_model)

        return self.out_proj(out)

    @torch.no_grad()
    def visualize_gbs_sampling(
        self,
        A: torch.Tensor,
        shots: int = GBS_CFG["default_shots"],
        c_ratio: float | None = None
    ) -> dict:  # type: ignore[return]
        """
        用 deepquantum MCMC 采样可视化子图检测（不用于训练，无梯度）。

        工作流程：
            1. 用 encode_graph_to_unitary_torch 得到 T、r
            2. 构建 deepquantum GaussianBosonSampling 电路
            3. 运行 MCMC 采样，返回 Fock 态计数字典
            4. 高频出现的 k-光子态对应图中的 k-团（dense subgraph）

        Args:
            A      : (N, N) CPU 邻接矩阵（2D tensor）。
            shots  : MCMC 采样次数。
            c_ratio: 可覆盖默认挤压强度（None 则使用 self.c_ratio）。

        Returns:
            result : {FockState: count} 字典（来自 deepquantum）。
        """
        import deepquantum as dq

        if c_ratio is None:
            c_ratio_t = self.c_ratio
        else:
            c_ratio_t = c_ratio

        T, r = encode_graph_to_unitary_torch(A.cpu(), c_ratio_t)

        gbs = dq.photonic.GaussianBosonSampling(
            nmode=self.nmode,
            squeezing=r,
            unitary=T.to(torch.cfloat)
        )
        gbs()
        gbs.detector = 'pnrd'
        result = gbs.measure(shots=shots, mcmc=True)
        # deepquantum 可能在 shots 少时返回 list[dict] 或 None
        if not isinstance(result, dict):
            result = {}
        return result  # type: ignore[return-value]
