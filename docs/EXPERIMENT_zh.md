# 实验协议与文件格式

## 数据用途

只从 GSM8K 原始 train 中划分下面四组，按 question 去重并排除与 test 相同的文本：

| 文件 | 默认大小 | 用途 |
| --- | ---: | --- |
| offline.jsonl | 256 | rollout、Jev 特征、单方向干预的监督训练样本；提前按 episode 划出约 20% Router validation |
| probe.jsonl | 64 | 每个 episode 用 ID 和 seed 确定 8 条，比较 baseline 与每个 candidate |
| guard.jsonl | 64 | 在线联合更新后的额外保留集检查 |
| online.jsonl | 128 | 在线更新的监督样本，与 offline/probe/guard 不重合 |
| test.jsonl | 原始 test 全部 | 只在最终冻结模型评测时使用；默认官方 GSM8K 为 1319 条 |

`prepare-data` 写 `manifest.json`，包含原始文件 SHA256、来源 revision、每个划分的哈希和大小。后续加载验证哈希，修改 JSONL 后不能继续用旧 manifest。所有 baseline/candidate 共享相同 probe 子集，不能给每个方向各抽一次题。

代码中的 probe 实质上是用于形成 Router 标签的 meta-training 数据，并非从未被使用的最终验证集。guard 被反复用于接受决策，也不是独立测试集。Router validation 与 Router train 的 episode ID 不重合，但它们共享 probe 池：因此 Router validation 只用于这套协议内的模型选择，不能代替最终 test 泛化结论。

如果使用不同种子重新划分，需要分别建立新数据目录、初始 adapter、干预目录和 Router。固定 test、提示模板和生成预算后再开展最终比较；不得根据 test 分数回头调阈值或学习率。

## 训练损失和代理指标

每条 JSONL 至少包含 `id/question/answer`。answer 来自 GSM8K 的正确解答，`<<算术标注>>` 被移除，保留可读解释与 `#### 数值`。训练只监督 assistant completion 和结束 token。超出 `max_length` 会明确报错，不截断标准答案。

默认 `reward=neg_nll`。输出日志中所有 delta 都是越大越好，负值代表更新更差。`reward=accuracy` 使用贪心生成和最终数值正确率；切换后必须重新收集。

`evaluate` 始终按相同的系统提示、`do_sample=False`、`num_beams=1`、`max_new_tokens` 比较。最终答案解析要求一行 `#### 数值`，允许合法千分位、负数和小数，按 Decimal 比较，`5` 与 `5.0` 等价。没有标记、单位、分数形式或科学计数法的最终答案按格式失败计错；每份报告有 `parse_failures`。该提示与解析协议属于本项目定义的 GSM8K 评测，不能直接横比使用其他 few-shot 提示或解析规则的排行榜数字。

## 特征协议

默认 83 个问题槽位：12 个全局 Noul、2 个五级 Score、1 个八类 Choice、60 个步骤 Noul、8 个 token Noul。无工具/memory 时不请求对应问题；步骤/token 不足时也不请求不存在的槽位。

维度计算：全局 Noul 12×3=36；Score 2×8=16；Choice 8+confidence+present=10；步骤 60×3=180；token 8×3=24；聚合统计 4。合计 **270**。

每个 Noul 保存 yes、no、present；每个 Score 保存完整有序分布、normalized_score、confidence、present；Choice 保存按预定义类别顺序排列的分布、confidence、present。缺失槽位全部为零且 present=0，而不是填入“否”的概率 1。

收到的有效问题缺答案、类型错误、NaN、概率不归一、未知类别都报错。允许服务省略零概率类别，但已返回的概率总和必须约等于 1。不同 schema 不能混训。完整 schema 保存在收集目录中。

Jev 缓存 key 包含 endpoint、请求模型、schema 和完整请求。缓存保存原始请求和响应，用于审计和节省重复调用；不会保存 API key。正式实验使用可访问的固定 Jev 版本，记录实际返回的 resolved model，收集和在线阶段发现版本不一致会拒绝继续。网络错误不会静默变成合成特征。

## Phase I 输出

```text
runs/phase1/
  contract.json
  feature_schema.json
  episodes/
    gsm8k-train-123.json
    ...
```

每条 episode 包含 `episode_id/split/contract_id/jev_features/delta_reward/baseline/scores/train_losses/probe_ids/trajectory/judge_meta/elapsed_seconds`。每条原子写入，重启跳过已完成记录。

所有 i 从初始 adapter 的同一状态出发。当前 release 每个收集目录只对应一个固定模型锚点；不是边收集边永久更新 Qwen。训练损失每个方向重复同样的单个监督样本、同样的步数和相同优化算法。

## 检查点兼容性

adapter 保存 U、V、R、K、rank、主干全部参数与缓冲区的哈希、目标模块路径、basis 哈希和 metadata。保存基而不是重新做 SVD，避免符号翻转或重根下基旋转导致 Router 的方向编号失去含义。

Router 保存网络与标准化参数、feature schema、收集合同、resolved Jev model、训练 record IDs 和决策超参数。在线阶段校验主干、目标模块、初始 R、basis、数据版本、Jev 模型/特征、学习率、干预步数、精度、生成预算等；不允许只换模型或学习率后继续使用旧 Router。

为确保一致性，每次加载模型会对冻结主干计算一次 SHA256。它需要读过全部模型张量，有少量启动开销，但不在每个干预中重复计算。检查点使用 tensor/基本 Python 类型并用 `weights_only=True` 读取。只加载可信来源的基础模型与本实验检查点。

## 在线恢复

```text
runs/online_router/
  adapter.pt
  events.json
  summary.json
```

`adapter.pt` 既包含当前参数，又包含已经完成的全部 episode 事件，是恢复进度的唯一事实来源。每个 episode 完成后原子替换；`events.json` 是方便查看的镜像。

用相同参数和输出路径重新执行 `adapt` 会加载它并继续；`--adapter` 始终传 Phase I 使用的原始锚点，程序自动找到在线目录的最新状态。改变 policy、Router、阈值、接受条件或数据会要求新目录。进程在一个 episode 中途退出时，最多重做该 episode；已保存的在线步骤不会重复应用。

每个 episode 都新建优化器，不跨 episode 保留 AdamW 动量。这是有意定义的实验算法，保证它与 Phase I 标签生成一致，也让回滚语义清楚。

## 结果与成本

每个最终报告保存所有 test 题的响应、正确性、解析成功状态、accuracy、Wilson 95% 区间、运行时间和 PyTorch 峰值 GPU allocated 显存（不是 nvidia-smi 的全部 reserved/进程占用）。`compare` 对相同题目作 2000 次配对 bootstrap，输出相对第一个报告的准确率差及 95% 区间。区间跨过 0 时不能仅凭点估计声称胜出；这个区间也没有涵盖不同训练随机种子的变异。

Phase I 每个 episode 默认成本：1 次 rollout、1 次 Jev 请求、K×steps 次反向训练，以及 (K+1)×probe_size 次验证前向。K=8、steps=2、probe_size=8 时是 16 次训练、72 次验证前向；不需要为 8 个方向各复制一个模型。

在线每次被选中的更新：steps 次反向、2×probe_size+2×guard_size 次验证；Jev Router 另需一次 rollout 和一次 API 调用。default guard_size=64 时在线验证可能比训练更耗时。先测 1–4 个 episode 的实际耗时，再估计完整实验成本。

## 扩展界面

现有 `QwenAgent.run` 是一次解题 rollout，没有工具、网页或外部 memory。扩展真实 agent 时，应将可见的 action/observation/tool/memory 轨迹放入 state；重新设计相关问题并升级 schema。不要让 Jev 对不存在的行为给出“有效性”。

`measure_interventions(model, adapter, loss_fn, eval_fn, settings)` 和 `adapt_with_guard(...)` 的 callable 接口可复用。需要保证 loss_fn 使用正确监督、eval_fn 不改参数、所有未选中的主干被冻结；自定义模型如果还有不在 state_dict/注册 buffer 中的可变缓存，需要在事务中显式纳入。不要把训练集标签或测试答案放入 Router 的状态。
