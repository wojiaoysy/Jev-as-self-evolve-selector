> 历史版本说明：此文档描述最初的 Router 实现；当前方法、运行模式与结果请以 [英文 README](README.md) 为准。

# Jev → Router → 正交子空间更新

面向 **AutoDL / RTX 4090 24GB / Ubuntu 22.04 / Python 3.10 / PyTorch 2.1.2 + CUDA 11.8** 的可复现实验项目。

它实现了附件方案的两阶段训练和在线接受/回滚，并修正了前向门控、Jev 特征解析、验证集泄漏和更新回滚的问题。默认模型为 **Qwen2.5-1.5B-Instruct**，默认 benchmark 为 **GSM8K**。

这是有标准答案监督的参数适应研究基线。Jev 为 Router 提供轨迹特征；Qwen 的梯度来自标准答案交叉熵。它不包含工具执行环境，也不声称已经验证了通用 agent 自进化能力或准确率提升。

建议按以下顺序阅读：

1. [方案审查](docs/REVIEW_zh.md)：哪些数学成立，哪些假设需要实验，为什么调整实现。
2. [AutoDL 完整操作流程](docs/AUTODL_zh.md)：环境安装、下载、冒烟、两阶段训练、在线更新、对照评测、恢复与排错。
3. [实验协议与文件格式](docs/EXPERIMENT_zh.md)：数据划分、收益定义、特征、检查点与复现边界。
4. [验证记录](docs/VALIDATION_zh.md)：已实际运行的检查及尚未验证的部分。

## 核心实现

| 模块 | 内容 |
| --- | --- |
| `src/jev_evolve/adapter.py` | 一层一个投影；固定 SVD 基；独立 FP32 `R_i`；所有历史更新持续参与前向 |
| `src/jev_evolve/intervention.py` | 公共起点干预；独立优化器；状态事务；联合 Top-k 更新与验证回滚 |
| `src/jev_evolve/judge.py` | 官方 Jev HTTP；83 个问题槽位；缺失掩码；固定 270 维特征；原始响应缓存 |
| `src/jev_evolve/router.py` | 每条样本独立归一化；回归、排序、稀疏损失；阈值拒绝更新 |
| `src/jev_evolve/model.py` | Qwen 加载；模板和答案掩码；分块词表损失；非重入梯度检查点 |
| `src/jev_evolve/data.py` | GSM8K 下载/本地导入；互斥数据划分；内容哈希；数值答案校验 |
| `src/jev_evolve/cli.py` | 命令行流水线、断点恢复、三种更新策略及不更新对照、配对统计 |

## 快速入口

在 AutoDL 终端，把项目放到 `/root/autodl-tmp/self-evolve` 后：

```bash
cd /root/autodl-tmp/self-evolve
bash scripts/setup_autodl.sh
source .venv/bin/activate
python scripts/smoke_cpu.py
python -m jev_evolve.cli --help
```

`smoke_cpu.py` 不下载模型，也不调用 Jev，使用随机小型 Qwen 和明确标记的合成特征，只检查工程链路。正式运行需要自行配置 `TYPESAFE_API_KEY`。完整命令见操作流程。

默认参数不是调参结论：K=8、r=8、Top-k≤2、单次 AdamW 2 步、lr=0.01、FP32 子空间、BF16 主干、序列长度 1024。每次最多训练 128 个标量；一个投影总共 512 个可训练标量。若标签全为零或没有正收益，先诊断实验，而不是把跑通当作有效。

本项目不会自动安装 FlashAttention、bitsandbytes、DeepSpeed 或升级已有 PyTorch。模型只从 safetensors 权重加载。
