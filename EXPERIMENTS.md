# 实验记录

## 这个分支是干嘛的

`lab-notes` 分支单独放实验结论、参数记录、踩坑经验。不跟代码混在一起，也不污染 `main` 历史。

## 怎么在本地同步这个目录

### 方式一：worktree（推荐）

```bash
# 在项目目录下执行
git worktree add ../2607-notes lab-notes
```

然后开一个新 VS Code 窗口打开 `../2607-notes`，笔记和代码互不影响。

### 方式二：clone

```bash
cd ..
git clone <仓库URL> 2607-notes
cd 2607-notes
git checkout lab-notes
```

# 0723
1. 一个是看看能不能用真机去跑
2. 其次是为了能用GBS做出的妥协（比如说 $Q \cdot K^{T}$ 变成酉矩阵），解释性问问觉得不够强，可能会导致loss很大
3. 概率幅尝试一下平方，还有