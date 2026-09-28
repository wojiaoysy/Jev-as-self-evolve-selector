# 实际验证记录

验证日期：2026-09-20。本记录区分工程验证与科学实验结果；没有把随机模型或合成特征当成 Jev/Qwen 的能力评测。

## 指定核心版本的 CPU 验证

本地独立环境：Windows、Python 3.10.20、PyTorch 2.1.2+cpu、Transformers 4.44.2、NumPy 1.26.4、tokenizers 0.19.1、huggingface-hub 0.25.2。PyTorch 核心版本和 Python 主次版本与目标匹配，操作系统及 CUDA build 与 AutoDL 不同。

实际完成：

- `python -m jev_evolve.cli preflight --cpu` 通过；`pip check` 无依赖冲突。
- `python -m unittest discover -s tests -v`：**22/22 通过**。
- `python scripts/smoke_cpu.py`：通过。小型随机 Qwen 真实反向传播、非重入 checkpoint、干预标签、Router 训练和接受更新正常；分块输出层 CE 与模型标准 completion-only loss 在规定容差内一致。
- `python scripts/smoke_pipeline.py`：通过。覆盖本地 safetensors 模型加载、初始化、部分收集及续跑、Router 拟合、random/all/router 策略、在线 checkpoint 续跑、同起点连续运行与续跑 R 逐值相等、最终报告与配对比较、错误合同拒绝。Jev 使用明确标记的 synthetic fixture，全程无真实评判请求。
- Python 全项目 compileall 与 `bash -n scripts/setup_autodl.sh` 通过。
- 本地 editable package 构建安装通过（使用已安装构建依赖、`--no-build-isolation`），命令行模块和 console entry point 可用。

22 项测试覆盖：增量矩阵顺序、零初始化、Frobenius 正交、仅选中参数变化、旧方向持续参与前向、干预顺序无关、参数/缓冲区/随机状态/模式回滚、异常与非有限数处理、拒绝和接受路径、空路由、更新范数约束、检查点一致性、API schema、缺失掩码、概率顺序、gold 不进入请求、认证错误处理、target 归一化、并列 ranking 可反向传播、答案解析、划分互斥和内容修改检测。

## 额外 GPU 冒烟

在本地 RTX 3060 Laptop GPU、PyTorch 2.13.0+cu126、Transformers 5.15.1 上运行小型随机 Qwen 的 BF16 主干 + FP32 adapter CUDA 冒烟通过。收益为负时实际触发回滚。它检查混合精度执行路径，**不能替代 4090 / CUDA 11.8 / torch 2.1.2 的实测**。

## 真实数据和 tokenizer 检查

下载并完整解析官方 GSM8K 原始文件：去重后的 train 7473 条，test 1319 条。源文件 SHA256：

```text
train 17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465
test  3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14
```

使用官方 Qwen2.5-1.5B-Instruct tokenizer、当前系统提示、seed=42 和默认 256/64/64/128 划分检查监督长度，结果：

| split | 样本数 | 最大 prompt+completion+结束 token 长度 |
| --- | ---: | ---: |
| offline | 256 | 347 |
| probe | 64 | 326 |
| guard | 64 | 356 |
| online | 128 | 389 |

均小于默认 max_length=1024。此检查没有用 test 答案调参。其他 seed、数据版本或自定义任务应重新运行 `check-data`。

## 尚未验证

- 未连接用户 AutoDL 实例，没有在 Ubuntu 22.04 / RTX 4090 24GB 上运行完整基础模型。
- 未下载或训练完整 Qwen2.5-1.5B 权重；未测量其目标环境峰值显存或吞吐。
- 未提供 TypeSafe 凭据，因此真实 Jev 服务调用、账户模型访问权限、配额与延迟未实测；HTTP 协议依据官方文档实现，parser 与错误路径已测试。
- 未进行完整 GSM8K 训练/测试，没有真实准确率提升结论。需按 AutoDL 操作流程运行四组实验后得到。

## AutoDL 上应完成的最后验证

先执行安装脚本、preflight、单元测试、CPU 冒烟、check-data 和 judge-smoke；再收集 4 个真实 episode，查看时间、显存、训练 loss 与 delta。确认后再扩到全量。默认超参数均是起点，不保证有正收益。

随机种子和固定样本用于降低随机差异；当前实现没有承诺不同 GPU、BLAS 或 CUDA 版本间的逐 bit 一致。小于数值精度的代理指标变化应通过重复计算或更高精度对照判断。
