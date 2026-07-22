import numpy as np
import torch
import deepquantum as dq
import pandas as pd

def encode_graph_to_unitary(A, c_ratio=0.3):
    """
    Encodes a given adjacency matrix into a unitary matrix suitable for Gaussian Boson Sampling (GBS).

    Args:
        A (np.ndarray):     The adjacency matrix of the graph.
        c_ratio (float):    The ratio of the scaling factor c to its maximum theoretical upper bound. 
                            Must be within the range (0, 1). Defaults to 0.9 for safety.

    Returns:
        T (np.ndarray):     The linear optical network matrix (a unitary matrix).
        r (np.ndarray):     A list of physical photon squeezing coefficients for each mode.
    """

    # Ensure the adjacency matrix is symmetric and has zeros on the diagonal
    assert np.allclose(A, A.T), "Adjacency matrix must be symmetric"
    assert np.all(np.diag(A) == 0), "Adjacency matrix must have zeros on the diagonal"

    W = A @ A.conj().T

    d, U = np.linalg.eigh(W)
    d = np.maximum(d, 0) 

    B = U.conj().T @ A @ U.conj()

    B_diag = np.diag(B)

    phi = np.angle(B_diag)/2.0

    phase = np.diag(np.exp(1j * phi))
    T = U @ phase @ U.conj().T

    sqrt_d = np.sqrt(d)
    max_sqrt_d = np.max(sqrt_d)

    if max_sqrt_d > 1e-9:
        c_max = 1.0 / max_sqrt_d
        c = c_ratio * c_max
    else:
        c = 1.0

    tanh_r = c * sqrt_d
    tanh_r = np.clip(tanh_r, 0, 0.999999)
    r = np.arctanh(tanh_r)
    r = torch.tensor(r, dtype=torch.float32)
    
    return T, r

def gbs_mean(df, mode):
    counts = torch.tensor(df['Count'].values, dtype=torch.float32)

    # ==================== 改造部分 ====================

    # 1. 将采样结果转换为形状为 (n_samples, mode) 的二值矩阵 (Tensor)
    # 每个状态如 '101100' 转换为 [1, 0, 1, 1, 0, 0]
    state_matrix = torch.tensor(
        [[int(bit) for bit in state] for state in df['State']],
        dtype=torch.float32
    )

    # 2. 计算每个状态的概率
    total_count = counts.sum()
    probs = counts / total_count

    # 3. 利用矩阵乘法一键生成图的邻接矩阵
    # state_matrix.T 的形状是 (mode, n_samples)，probs.diag() 是 (n_samples, n_samples)，state_matrix 是 (n_samples, mode)
    # 相乘后的形状刚好是 (mode, mode)，即两两节点同时出现的概率和
    adj_matrix = torch.matmul(state_matrix.T, torch.matmul(probs.diag(), state_matrix))

    # 4. 移除自环权重
    adj_matrix.fill_diagonal_(0)

    return adj_matrix

def gbs_theory(T, r):
    """
    理论计算GBS的期望邻接矩阵（不使用MCMC采样）

    Args:
        T (np.ndarray):     The linear optical network matrix (a unitary matrix).
        r (np.ndarray):    A list of physical photon squeezing coefficients for each mode.

    Returns:
        adj_matrix:         理论期望的邻接矩阵 (mode, mode)
    """
    # 将r转换为numpy数组（如果是tensor）
    if isinstance(r, torch.Tensor):
        r = r.cpu().numpy()

    # 期望光子计数的方差 = sinh^2(r)
    var = np.sinh(r) ** 2

    # 使用酉矩阵T来计算期望的二阶关联
    # 简化：期望邻接矩阵 = |T|^2 * 平均方差
    T_abs_sq = np.abs(T) ** 2
    avg_var = np.mean(var)

    # 邻接矩阵 = |T|^2 * 方差
    adj_matrix = T_abs_sq * avg_var

    # 归一化，使每行和为1
    np.fill_diagonal(adj_matrix, 0)
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    adj_matrix = adj_matrix / row_sums

    return adj_matrix

def gbs_sample(T, r, mode, shots):
    """
    Simulates Gaussian Boson Sampling (GBS) given a unitary matrix and squeezing parameters.

    Args:
        T (np.ndarray):     The linear optical network matrix (a unitary matrix).
        r (torch.Tensor):   A list of physical photon squeezing coefficients for each mode.
        mode (int):         Number of modes (GBS channels), equals seq_len.
        shots (int):        Number of samples to generate.

    Returns:
        samples (list):     A list of samples, where each sample is a list of photon counts for each mode.
    """

    gbs = dq.photonic.GaussianBosonSampling(nmode=mode, squeezing=r, unitary=T)
    gbs()
    gbs.detector = 'pnrd'
    result = gbs.measure(shots=shots, mcmc=True)
    formatted_data = {str(k).strip('|>'): v for k, v in result.items()}

    df = pd.DataFrame(list(formatted_data.items()), columns=['State', 'Count'])

    adj_matrix = gbs_mean(df, mode)

    return adj_matrix