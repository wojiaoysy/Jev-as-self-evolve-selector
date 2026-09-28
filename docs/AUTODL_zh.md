# AutoDL RTX 4090 24GB 完整操作流程

以下命令在 **AutoDL Ubuntu 22.04 终端**执行。已有环境应为 Python 3.10、PyTorch 2.1.2、CUDA 11.8。默认使用一张 4090；不需要多卡。

## 1. 上传项目并检查机器

将整个项目文件夹上传或解压到 `/root/autodl-tmp/self-evolve`。不要上传本地 `.venv-test`、`__pycache__`、模型缓存或旧实验输出。若使用项目提供的 zip，里面已排除这些文件。

```bash
cd /root/autodl-tmp/self-evolve
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
df -h /root/autodl-tmp
```

建议预留至少 15GB 磁盘，并保留足够系统 RAM 供模型加载和单矩阵 SVD 使用（例如 16GB 或更多）。这不是额外的 GPU 显存要求。不要把模型和实验数据写到容量较小的系统盘；路径按你的实例挂载情况调整。

## 2. 安装固定依赖，保留已有 PyTorch

```bash
bash scripts/setup_autodl.sh
source .venv/bin/activate
```

脚本会验证已有 torch 版本，用 `--system-site-packages` 创建项目虚拟环境，复用镜像中的 torch，只在 `.venv` 内安装项目依赖，然后执行环境检查和单元测试。不要运行不带版本约束的 `pip install -U torch transformers`。

手动执行时对应：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m jev_evolve.cli preflight
python -m unittest discover -s tests -v
```

`preflight` 期望 Python 3.10、torch 2.1.2、torch 自带 CUDA runtime 11.8、Transformers 4.44.2、NumPy 1.x，以及可用的 CUDA/BF16。`nvidia-smi` 显示的 CUDA Version 是驱动能力，可能与 `torch.version.cuda` 不同；本项目检查后者。不要只因为 nvidia-smi 显示其他版本就重装驱动。

如果镜像实际没有 torch 2.1.2，优先重新选择题述镜像；确需在干净环境安装时，官方命令为：

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

上面的补装不是正常流程的必需步骤。项目不使用 torchvision/torchaudio，也不需要 FlashAttention 编译。

## 3. 完全离线的工程冒烟

```bash
python scripts/smoke_cpu.py
```

它随机构造一个两层小 Qwen，真实运行 tokenizer、completion loss、非重入 checkpoint、子空间干预、Router 训练和在线接受判断，输出 `status: PASS`。无需下载模型、Jev key 或联网。

输出会标记 `SYNTHETIC-NOT-JEV`。它只是工程验证，里面的“收益”来自随机小模型，不能作为实验结果。正式实验不会自动退回这种合成模式。

## 4. 下载 Qwen 并建立本地配置

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
python -m jev_evolve.cli download-model \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output models/qwen2.5-1.5b-instruct
```

命令先解析具体 commit，再下载该版本的 safetensors 和 tokenizer 文件；commit 写在 `download_manifest.json` 中。下载失败可原命令重试。若实例无法访问 Hugging Face，可先在有访问条件的机器运行下载命令，再把完整目录上传到这里；不要只上传一个权重分片。

生成本地运行配置：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('configs/autodl_4090.json')
c = json.loads(p.read_text())
c['model']['name'] = str(Path('models/qwen2.5-1.5b-instruct').resolve())
Path('configs/local.json').write_text(json.dumps(c, indent=2), encoding='utf-8')
PY
```

后续统一用 `--config configs/local.json`。如果换用其他路径，修改 `model.name` 即可；权重哈希仍用于检查是否为同一模型。

## 5. 下载和划分 GSM8K

```bash
python -m jev_evolve.cli prepare-data \
  --output data/gsm8k --seed 42 \
  --offline 256 --probe 64 --guard 64 --online 128
```

该命令从 GSM8K 原作者公开仓库获取 train/test JSONL，并固定划分。若已有原始文件，把文件命名为 `train.jsonl` 和 `test.jsonl`，用：

```bash
python -m jev_evolve.cli prepare-data \
  --source-dir /root/autodl-tmp/gsm8k_raw \
  --output data/gsm8k --seed 42 \
  --offline 256 --probe 64 --guard 64 --online 128
```

两种方式二选一，数据目录非空时不会覆盖。保留 `manifest.json` 和划分文件，跨机器复制这一份数据可保持完全相同的样本顺序。更严格的来源复现可以通过 `--source-revision <仓库commit>` 固定源版本，同时记录原始文件 SHA256。

先不对 test 做调参。offline 内预分 Router train/val，probe 生成标签，guard 做在线额外检查，online 提供后续监督样本，最终 test 独立保存。

下载 tokenizer 后，可在加载大模型前检查所有开发/训练样本是否符合长度预算：

```bash
python -m jev_evolve.cli check-data --config configs/local.json --data data/gsm8k
```

检查只读取 offline/probe/guard/online 的监督长度，不读取 test 标准答案。若超限，在正式初始化和收集前修改 `model.max_length`。

## 6. 配置 Jev 访问并验证响应

在 TypeSafe 控制台准备可用 API key，在终端安全输入：

```bash
read -r -s -p 'TypeSafe API key: ' TYPESAFE_API_KEY
echo
export TYPESAFE_API_KEY
python -m jev_evolve.cli judge-smoke --config configs/local.json
```

这一步会向配置的官方接口发送一条简单算术轨迹和问题，并可能产生服务费用。key 不写入配置、缓存或检查点，也不需要发送到聊天里。

成功时显示 feature_dimension=270、schema ID、resolved model 和 usage。默认请求 `jev-1.13.0`，以官方文档示例版本为起点；如果账户不支持此版本，应根据你的可用模型选择固定版本，在正式收集前修改 `judge.model` 并重新执行冒烟。不要在收集一半时从固定版本切换到 `jev-latest`。

401/403：检查账户权限、key 和 endpoint；429：检查额度及限流；5xx 或超时：有限重试后退出，可稍后重跑。出错时不会使用假分数。cache 目录保存完整请求和原始响应，因此也包含题目与模型回答。

## 7. 初始化单层适配器

```bash
python -m jev_evolve.cli init \
  --config configs/local.json \
  --output runs/initial.pt
```

默认目标 `model.layers.14.self_attn.q_proj` 是零基索引 14，即第 15 个 decoder block 的 query 投影。程序冻结模型，计算基础权重哈希，在 CPU 用 FP32 对这一矩阵做一次 SVD，然后保存 U/V 和全零 R。

输出应显示 `R_parameters=512`。以后始终加载这个检查点的基，不要给每个阶段重新 SVD。已存在的 `initial.pt` 不会被覆盖。

## 8. 小规模真实实验冒烟

先收集 4 条真实 episode：

```bash
python -m jev_evolve.cli collect \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt --output runs/phase1 \
  --max-episodes 4
```

每条将运行 Qwen、Jev、baseline 和 8 个独立子空间干预。输出包含最大 delta 和耗时。默认第 1 条是 Router validation，随后几条是 train，所以短冒烟前缀也能覆盖两类。数据很少，只用于验证工程。

建议此时检查：

```bash
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('runs/phase1/episodes').glob('*.json')):
    r = json.loads(p.read_text())
    print(r['episode_id'], r['split'], r['delta_reward'], r['elapsed_seconds'])
PY
nvidia-smi
```

如果 delta 全部等于 0、全部是极小数、训练 loss 不变或所有 episode 最佳方向完全相同，先看后面的诊断说明。不要直接把实验扩大到数千条。

可在独立输出路径上检查 Router 和在线循环：

```bash
python -m jev_evolve.cli train-router \
  --config configs/local.json --records runs/phase1 \
  --output runs/router_smoke.pt
python -m jev_evolve.cli adapt \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt --router runs/router_smoke.pt \
  --policy router --output runs/online_smoke --max-episodes 2
```

若 Router 没有选出方向，`no_positive_route` 是允许的正常结果，不应强制训练来制造进度。

## 9. Phase I：完整干预数据

```bash
python -m jev_evolve.cli collect \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt --output runs/phase1
```

它会跳过上一步完成的 4 条，继续到全部 256 条。每个 episode 原子保存，网络或进程中断后使用同一命令恢复。相同轨迹和问题可复用 Jev 缓存。

收集过程中不会累积改变 Qwen；全部训练标签均相对同一 `initial.pt` 模型锚点产生。若改变目标层、学习率、训练步数、收益指标、精度或特征，使用新的配置、初始 checkpoint、收集目录与 Router，不要混入原目录。

## 10. Phase II：训练 Router

```bash
python -m jev_evolve.cli train-router \
  --config configs/local.json --records runs/phase1 \
  --output runs/router.pt
```

Router 在 CPU 训练，默认 100 epoch，按 Router validation loss 选择最佳状态。产物：

- `runs/router.pt`：网络、标准化、特征协议、数据和模型合同。
- `runs/router.pt.metrics.json`：训练/验证 episode 数、正收益比例、收益并列比例、最佳方向分布、每轮 validation loss、已选择方向的正收益比例及 best-single regret。

`mean_best_single_regret` 只比较已选集合包含的最佳**单方向**收益与全体最佳单方向收益，不能当成联合更新的实际 regret。所有标签为负时会警告；Router 跑完不代表学到了有用策略。

## 11. 在线阶段与对照组

所有策略使用同一初始 checkpoint、online 样本顺序、监督损失、probe、guard、优化器和步数，各自写独立输出目录：

```bash
python -m jev_evolve.cli adapt \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt --router runs/router.pt \
  --policy router --output runs/online_router

python -m jev_evolve.cli adapt \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt \
  --policy random --output runs/online_random

python -m jev_evolve.cli adapt \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt \
  --policy all --output runs/online_all
```

不更新组直接使用 `runs/initial.pt` 评测即可，无需另外执行 `adapt --policy none`。

Router 不再遍历 K 个单方向干预：只预测一次 Top-k，然后对它们联合训练一次。candidate 在 probe 上严格改善且 guard 不退化时保留，否则回滚。`events.json` 保存选择方向、alpha、before/after、接受结果和耗时。`adapter.pt` 保存当前完整增量与进度。

中断恢复：原命令直接重跑。`--adapter` 仍写 `runs/initial.pt`，程序自动从对应在线目录的 `adapter.pt` 恢复。要重做一个策略实验，请换输出目录；不要把已演化 checkpoint 当成同一旧 Router 的初始锚点。

需要刷新 Router 时，将新的在线 adapter 作为新的 Phase I 锚点，收集到新目录、训练新 Router，再从这个新锚点在线更新。应使用新收集任务/预先设计的轮次协议，防止不断重用同一小集合产生过拟合。

## 12. 冻结所有模型，在同一 test 上最终比较

确定超参数后再执行以下全量评测。各组配置的提示、token 预算、精度和数据必须保持一致。

```bash
python -m jev_evolve.cli evaluate \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/initial.pt --label base \
  --output runs/eval_base.json

python -m jev_evolve.cli evaluate \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/online_router/adapter.pt --label router \
  --output runs/eval_router.json

python -m jev_evolve.cli evaluate \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/online_random/adapter.pt --label random \
  --output runs/eval_random.json

python -m jev_evolve.cli evaluate \
  --config configs/local.json --data data/gsm8k \
  --adapter runs/online_all/adapter.pt --label all \
  --output runs/eval_all.json

python -m jev_evolve.cli compare \
  --reports runs/eval_base.json runs/eval_router.json runs/eval_random.json runs/eval_all.json \
  --output runs/comparison.json
```

`compare` 第一个报告是参考组。报告含准确率、相对 base 的百分点差（文件中用 0–1 小数表示）、配对 bootstrap 区间；不兼容的题目、基或生成协议会拒绝比较。

若只想验证评测命令，可为所有组加相同 `--limit 20` 并用 `runs/debug_eval_*.json`。结果会标记 `full_test=false`，不能当成全量 benchmark 结果，也不要依据这 20 道 test 题调参。正式报告使用不带 limit 的命令。

## 13. 调参、显存与标签排错

| 现象 | 先检查什么 | 处理方式 |
| --- | --- | --- |
| torch/transformers 版本不符 | 是否激活 `.venv`，是否使用 `python -m pip` | 重新执行固定依赖安装与 preflight，不升级 torch |
| NumPy 初始化错误 | 是否装了 NumPy 2.x | 在项目环境重装 `numpy==1.26.4` |
| CUDA OOM | 是否有其他进程，实际 max_length，是否关闭 checkpoint | 保持单样本训练；先缩短测试配置的序列/生成长度；新建实验输出 |
| 样本超过 max_length | 报错中有 ID 与所需 token 数 | 提高 max_length 并重新建立一致的实验合同；不要删除 gold 末尾 |
| delta 全零 | BF16 舍入、lr 太小、accuracy 样本太少 | 保持负 NLL 先验证；检查训练 loss；在新实验中增大学习率/步数或 FP32 对照 |
| delta 几乎全负 | 当前方向或数据不适合此更新 | 比较 lr、目标层、r，检查训练目标；允许不更新 |
| 最佳方向总相同 | SVD 方向敏感度不均 | 增加固定方向基线；检验 Router 是否只是学常量 |
| Router 全不更新 | 标签质量、阈值、样本不足 | 只依据 Router validation 调阈值；保存新 Router 并使用对应配置 |
| 在线大部分拒绝 | 联合干预非线性、guard 过严格、分布漂移 | 从 Top-k=1 做控制实验；检查 guard 分数与日志；新实验重新采标签 |
| NLL 改善但准确率下降 | 代理目标与生成能力不一致 | 如实报告；尝试 accuracy 标签/接受指标、新验证集设计 |
| 速度慢 | guard 有 64 条且前后各评估 | 先测单 episode；较小 guard 只能用于早期探索，不得伪装为同等可靠结论 |
| Jev 缺字段/返回类型变化 | cache 内原始响应 | 停止混用，更新 parser 与 schema 后重新采样，不静默填零 |
| 收集目录/Router 合同不符 | 变更过配置、checkpoint、数据 | 新建独立实验；恢复时用原参数 |

默认每个 episode 的 `max_update_norm=1.0` 是增量权重 Frobenius 范数上限。改变它会改变干预算法，需重新收集。主干冻结并不意味着反向传播不占显存；梯度必须穿过适配器后的 decoder 层。

默认对第 15 层 q_proj 进行一次实验。比较其他层时只修改 `model.target_module`，例如 `model.layers.7.self_attn.q_proj` 或 `model.layers.21.self_attn.q_proj`，每层从原始 Qwen 分别初始化、分别收集和训练 Router。不要把第 7 层的 Router 应用于第 21 层。层选择应依据独立开发评估协议，不用最终 test 挑层。

## 14. 需要保存的实验材料

保存项目版本、最终 config、基础模型下载 manifest、数据 manifest 与全部 split、initial.pt、Phase I records/schema/contract、Router 及 metrics、在线每组 adapter/events/summary、所有 eval 报告和 comparison。Jev 缓存可用于复核问题与返回分布。

记录至少三个随机种子的结果、正收益标签比例、Router 选择分布和接受率；算力有限时先完成一个种子的四组对照并标记探索性结果。最小成功标准是管线可靠、无数据泄漏、对照公平、结果可重现；准确率是否提高由真实评测决定。
