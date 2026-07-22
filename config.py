# config.py
import torch

class Config:
    # 数据集相关参数
    c_in = 22          # 输入特征数
    seq_len = 48      # 历史滑动窗口长度（也是GBS上下文窗口长度mode）
    pred_len = 16      # 预测未来长度
    use_diff = False    # 是否使用一阶差分

    # 数据文件配置
    data_path = "data/weather.csv"  # 数据文件路径
    train_start = 0         # 训练数据起始索引
    train_end = 3000       # 训练数据结束索引（用3000条足够训练）
    val_start = 3000      # 验证数据起始索引
    val_end = 3500         # 验证数据结束索引
    test_start = 3500      # 测试/推理数据起始索引
    test_end = None         # 测试/推理数据结束索引（None表示到数据末尾）

    # 模型架构参数
    d_model = c_in      # 隐藏层维度（通常设置为c_in）
    n_heads = 22       # 注意力头数（确保 d_model 能被 n_heads 整除）
    d_ff = 256        # FFN 中间层维度（通常为d_model的4倍）
    n_layers = 2      # TransformerBlock 的堆叠层数
    dropout = 0.1     # 防止过拟合的随机失活率
    mlp_layers = 3     # FFN中额外的MLP层数

    # GBS模块参数
    use_gbs = False  # True: GBS量子采样(dq), False: GBS理论计算, "standard": 标准Attention
    compare_standard = True  # 是否同时运行标准Attention进行对比
    shots = 1024      # GBS模块采样的次数（仅当use_gbs="gbs"时有效）

    # 输出文件夹
    output_dir = "output"
    model_path = "best_quantum_transformer.pth"  # 模型保存路径

    # 训练超参数
    batch_size = 32
    lr = 1e-4        # 经典的 AdamW 初始学习率
    epochs = 30         # 训练轮数
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 兼容性别名
    c_out = c_in