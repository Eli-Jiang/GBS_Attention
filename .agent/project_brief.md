# GBS-Based Quantum Attention Mechanism — Project Brief (Rev 3)

> 更新时间：2026-07-12（第三版，基于 precode.ipynb 深度解读）
> 状态：**传统基线完成（需重构） / GBS 注意力机制待实现**

---

## 1. Precode (`precode.ipynb`) 完整解读

### 1.1 整体流程

```
随机/计算得到的邻接矩阵 A (N×N, 对称, 对角线为0)
  ↓ encode_graph_to_unitary(A, c_ratio=0.3)
  ↓ → 干涉仪酉矩阵 T (N×N), 挤压参数 r (长度 N)
  ↓ GaussianBosonSampling(nmode=6, squeezing=r, unitary=T)
  ↓ gbs.detector = 'pnrd'
  ↓ gbs.measure(shots=2048, mcmc=True)
  ↓ 后处理：过滤 / 二值化 k-光子态 → 子图检测
```

### 1.2 核心编码函数 `encode_graph_to_unitary`

这是整个 GBS 注意力机制的数学基础，实现了从**图的邻接矩阵到 GBS 参数**的严格映射：

```python
def encode_graph_to_unitary(A, c_ratio=0.3):
    # A: 对称矩阵，对角线全 0
    W = A @ A.conj().T                   # W = A²（格拉姆矩阵，PSD）
    d, U = np.linalg.eigh(W)             # 特征值分解，d≥0
    d = np.maximum(d, 0)

    B = U.conj().T @ A @ U.conj()        # 对角化 A（近似）
    phi = np.angle(np.diag(B)) / 2.0    # 提取相位
    T = U @ np.diag(np.exp(1j*phi)) @ U.conj().T  # 干涉仪酉矩阵

    # 将特征值映射为挤压参数 r
    sqrt_d = np.sqrt(d)
    c = c_ratio / np.max(sqrt_d)        # 缩放因子（保证 tanh(r) < 1）
    tanh_r = np.clip(c * sqrt_d, 0, 0.999999)
    r = np.arctanh(tanh_r)
    return T, r
```

**物理含义**：GBS 的采样概率正比于图子矩阵的 **Hafnian**（完美匹配数）。检测到某个 k 光子模式 S，意味着 S 对应的 k 个节点构成一个**高度密集的子图**（接近 k-团）。这给了 GBS 天然的**高阶图注意力**能力——不仅检测节点对（2阶），还能检测三角关系、4-团等多体关联。

### 1.3 后处理：光子数分组 → 注意力信号

| 光子数 k | 物理含义 | 对应注意力语义 |
|---|---|---|
| 0 (全零) | 真空态，无信息 | 过滤掉 |
| 2 | 2个节点共现 | **二元关系强度** → $N \times N$ 注意力矩阵 |
| 3 | 3个节点共现 | **三元关系强度** → 高阶依赖 |
| 4 | 4个节点构成子团 | **四体关联** → 全局结构 |

**二值化策略**（Cell 20）：将光子计数 $>1$ 的通道也计为 1，消除多光子冲突的噪声影响，恢复其对应的图拓扑信息。

---

## 2. GBS 注意力机制的正确设计

### 2.1 时序注意力的应用方式

对于长度 $L$ 的时序输入（如 96 时间步），先做 **Patch 化**：
$$\text{patch\_size} = P, \quad N = L / P \quad (\text{e.g.} \ P=16, N=6)$$

每个 Patch 压缩为一个 $d_k$ 维向量，N 个 Patch 构成 N 个 Token。

计算 Token 间的相似度矩阵：
$$S_{ij} = q_i \cdot k_j^T / \sqrt{d_k}$$

对称化（因为 GBS 要求对称邻接矩阵，对角线为 0）：
$$A_{ij} = \begin{cases} (S_{ij} + S_{ji})/2 & i \neq j \\ 0 & i = j \end{cases}$$

然后调用 `encode_graph_to_unitary(A)` → GBS → 提取 2-光子概率矩阵 $\hat{A}^{(2)}_{ij}$（第 $i,j$ 模各有 1 个光子的概率）作为新的注意力分数 → softmax → 加权 V。

### 2.2 Open 问题（当前未解决）

| 问题 | 分析 |
|---|---|
| `encode_graph_to_unitary` 中用的是 `np.linalg.eigh`（numpy），不可微 | 若要训练 QKV 矩阵需要改用 `torch.linalg.eigh`，或对编码函数走 STE |
| `gbs.measure(shots=2048)` 随机采样不可微 | 可改用 `gbs(is_prob=True)` 获取解析概率字典（无需采样），然后手动聚合 2-光子概率 |
| N=6 模，二光子概率对应 $\binom{6}{2}=15$ 个节点对 | 需要将 15 个概率重组为 6×6 的对称注意力矩阵（对角线置 0） |
| `c_ratio=0.3` 是硬编码参数 | 挤压强度影响 GBS 采样分布，可设为可学习标量 |

---

## 3. 【重要】传统基线的改进方案

### 3.1 现有基线的问题

现有的 `classical_baseline.py` 做的是：
```
输入: (B, 96, 7) → Linear/LSTM → 预测: (B, 24, 7)
```

这与 GBS 注意力机制**根本不对应**：
1. GBS 注意力工作在 **N=6 个 Patch** 上，而非 96 个时间步上
2. Linear/LSTM 不包含注意力模块，无法作为"注意力机制对比"的基线
3. ETTh1 太简单（线性强，周期清晰），无法展示多体关联的优势

### 3.2 应该怎么改：架构对齐

**正确的对照基线**应该与 GBS 注意力有**相同的 Patch 化前处理**，只在注意力模块上有区别：

```
输入: (B, L, C)
  ↓ Patch 化: patch_size=P → N = L/P 个 Patch
  ↓ Linear Embedding: 每个 Patch → d_model 维向量
  ↓ 注意力层（对比点！）
    ├── 基线: 标准 Self-Attention (QKV + Softmax)
    └── 量子: GBS-Attention (QKV → 对称化 → encode_graph → GBS 概率)
  ↓ 预测头: d_model → pred_len * C
```

推荐参数：`patch_size=16, N=6 (96/16=6), d_model=32, pred_len=24`

### 3.3 应该怎么改：数据集扩充

仅用 ETTh1 不足以论证量子优势边界，需要覆盖三类特性：

#### ① 高维多变量（High-dimensional Multivariate）
**推荐**：`Weather` 数据集（21 个气象变量，小时采样，约 52696 行）
- 特点：变量数多（21 维），特征间有复杂的物理相关性（温度/气压/湿度/风速等）
- 下载：[原始 GitHub](https://github.com/thuml/Autoformer/tree/main/dataset)（`weather.csv`）
- 意义：GBS 的多体关联应当在高维异质特征间更有优势

#### ② 非线性 / 强突变（Non-linear / Non-stationary）
**推荐**：`Exchange Rate` 数据集（8 种货币，日采样，约 7588 行）
- 特点：汇率受政策/事件驱动，非线性强，难以用线性模型捕获
- 下载：[Autoformer 数据包](https://github.com/thuml/Autoformer/tree/main/dataset)（`exchange_rate.csv`）
- 意义：如果 GBS 能比 Softmax 更好捕获汇率间的非线性关联，说明量子优势存在

#### ③ 多源异构（Multi-source Heterogeneous）
**推荐**：`ILI`（流感样症状数据，8 个变量，周采样，约 966 行，多机构来源）
- 特点：来源异构（不同医疗机构），季节性 + 偶发性事件（大流行），样本少
- 下载：[Autoformer 数据包](https://github.com/thuml/Autoformer/tree/main/dataset)（`national_illness.csv`）
- 意义：少样本下量子方法的泛化能力测试

### 3.4 具体改动列表

| 改动项 | 现有实现 | 应改为 |
|---|---|---|
| 数据预处理 | 直接 sliding window (96,7) | 先 patch，再 sliding window：(6 patches, d_model) |
| 基线模型 | Linear / LSTM | **PatchTransformer**（标准 Softmax Self-Attention，N=6 tokens） |
| 数据集 | ETTh1（1个） | ETTh1 + Weather + Exchange Rate + ILI（4个） |
| 评估指标 | MSE, MAE, R², per-feature | 同上，额外加 MAPE（平均绝对百分比误差）和预测误差随预测步数的曲线 |
| 训练配置 | epochs=10 | epochs=20~30，加 early stopping，更好反映收敛性 |

---

## 4. 已完成实验

| 模型 | 数据集 | 设置 | Test MSE | Test MAE | $R^2$ |
|---|---|---|---|---|---|
| Linear | ETTh1 | seq=96, pred=24 | 0.3491 | 0.4026 | 0.7107 |
| LSTM | ETTh1 | seq=96, pred=24 | 0.5135 | 0.5034 | 0.5749 |

> 注：以上结果仅用于**证明 ETTh1 数据集的可预测性**，架构与 GBS 注意力不对应，不能直接用于对照。

---

## 5. 修订后的完整子任务规划

| 步骤 | 内容 | 状态 |
|---|---|---|
| **1A** | ETTh1 数据集 + Linear/LSTM 验证可预测性 | ✅ 完成 |
| **1B** | 下载 Weather / Exchange Rate / ILI 数据集 | 📋 待做 |
| **2A** | 重构数据预处理：加入 Patch 化（patch_size=16, N=6） | 📋 待做 |
| **2B** | 实现 PatchTransformer 经典基线（标准 Softmax Attention，N=6 token） | 📋 待做 |
| **2C** | 在 4 个数据集上评估经典基线，形成完整的"经典方法可预测"证明 | 📋 待做 |
| **3A** | 实现 `GBSAttentionLayer`：`encode_graph_to_unitary` PyTorch 化 + `is_prob=True` 解析概率 | 📋 待做 |
| **3B** | 端到端训练 GBS-Attention vs Softmax-Attention | 📋 待做 |
| **3C** | 界定量子注意力"优势边界" | 📋 待做 |

---

*最后更新：2026-07-12。基于 precode.ipynb 全文解读生成。*
