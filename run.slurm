#!/bin/bash
#SBATCH --job-name=my_test        # 任务名称
#SBATCH --nodes=1                 # 节点数
#SBATCH --ntasks-per-node=1       # 任务数
#SBATCH --cpus-per-task=4         # CPU核心数
#SBATCH --time=01:00:00           # 运行时间
#SBATCH --partition=64c512g       # 队列名称，这里以思源一号CPU队列为例

# 加载环境
module load python
source myenv/bin/activate

# 切换到代码目录并运行
python train.py
python inference.py