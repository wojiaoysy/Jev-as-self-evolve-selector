# 方案审查与本版取舍

结论：可以作为“诊断特征是否有助于选择参数更新方向”的可检验研究方案。正交参数化成立，干预数据训练 Router 的流程也能落地；但“正交方向对应能力”“诊断足以预测梯度收益”“回滚保证不会遗忘”都不是已经成立的结论。

## 1. 保留的数学结构

对冻结权重 W 做 SVD，从左右奇异向量中各取 Kr 列，并按 r 列分组。定义：

\[
\Delta W_i=U_iR_iV_i^\top,\quad W'=W+\sum_{i=1}^K\Delta W_i.
\]

如果 i≠j，则：

\[
\langle\Delta W_i,\Delta W_j\rangle_F
=\operatorname{tr}(V_iR_i^\top U_i^\top U_jR_jV_j^\top)=0.
\]

因此不同子空间在权重的 Frobenius 几何下正交。对于全部方向的有限更新：

\[
\|\delta W\|_F^2=\sum_i\|\delta R_i\|_F^2.
\]

实现用这个等式约束每个 episode 的总更新范数。`nn.Linear` 使用行向量约定，增量必须计算为 `x @ V_i @ R_i.T @ U_i.T`。

只截取奇异向量来构造增量基，**原始 W 完整保留**；没有把模型压缩成一个 Kr 秩模型。`R_i` 零初始化时，适配器前向与原投影相同。SVD 只对选中的一个投影执行一次，基保存在检查点中，后续直接加载。

这也不是把整个权重空间完整分成 K 份：这里只允许匹配块 `U_i R_i V_i^T`，不包含 i≠j 的交叉块。因此自由度是 K×r²，而不是 (Kr)²，也不是 d_in×d_out。限制很强，表达能力不足本身就是需要实验排查的原因。

## 2. 关键修正

### 2.1 将更新选择与前向计算分离

附件中只累加当前 active 的方向，会在路由切换时撤下先前已学到的方向。即使参数还在，模型行为也会突然改变，baseline 与 candidate 就可能不再可比。

本版始终使用全部 K 个方向的已保存参数进行前向；Router 只决定本次哪些独立 `Parameter` 能接收梯度并进入优化器。默认不将 alpha 乘入前向，也不将它乘入学习率。这样 Phase I 与在线阶段使用同一种更新算法。

如果未来增加 alpha 学习率缩放，必须重新生成相应干预数据并训练 Router；否则单方向标签对应的是固定学习率，而部署执行的是另一种干预。

### 2.2 定义真正可微的损失

Jev HTTP 输出的概率不是 Qwen 计算图的一部分。把它直接拼成标量并 `backward()`，不能获得 Qwen 参数梯度。

本版使用 GSM8K **训练数据**中的标准解答，优化 completion-only teacher-forced 交叉熵。题目和系统提示的 token 全部 mask 为 -100；监督目标包括解释和最终答案。Qwen rollout 用于 Jev 特征，不被当成自动正确的训练标签。

因此本版属于“有监督的在线参数适应”，不等于无标注的自主强化学习。扩展到真实 agent 需要另外定义可学习信号，例如可验证成功轨迹、纠正后的示范，或明确实现策略梯度；仅仅加更多 Jev 问题不能替代这一步。

### 2.3 用完整事务保证公共起点

每个方向都从相同 R、相同固定基、相同验证样本、相同模型缓冲区与随机状态开始，并新建优化器。干预结束不保留候选参数。异常、NaN 和拒绝都会触发恢复。

主干权重被冻结且经身份哈希验证，不复制一份 1.5B 模型到 GPU。事务只备份 R 和注册的模型缓冲区，并恢复训练模式和 requires_grad 标记。清除梯度，避免上一次计算残留影响下一次。

这一控制实验测量的是**在当前权重、给定监督损失、给定步长和样本上的局部干预效果**，不是对某种认知能力的因果定位。

### 2.4 对齐收益方向，保留真正的最终指标

统一约定 J 越大越好。默认：

\[
J=-\frac1{|B|}\sum_{x\in B}\mathrm{NLL}_{\text{completion}}(x),
\quad\Delta J_i=J_i-J_0.
\]

每个任务先对 completion token 求均值，再对任务求均值，因此较长解释不会单纯因长度而占更大权重。

默认用负 NLL，是因为只有 8 个 probe 样本时，准确率只能按 12.5 个百分点跳变，极小参数更新可能全部测成 0；负 NLL 能提供更细的监督信号。但它是代理目标，**不是 GSM8K 准确率，也不保证准确率提高**。

可把 `intervention.reward` 改成 `accuracy`，让标签和接受条件直接使用生成后的精确答案准确率；这通常更慢、标签更稀疏，必须新建收集目录、重新收集和训练。最终 `evaluate` 始终测同一份 GSM8K test 的生成准确率。

### 2.5 接受/回滚的边界

联合更新 Top-k 后，在当前 probe 子集和另一份固定 guard 集上检验：

\[
J_{\rm probe}(\theta')-J_{\rm probe}(\theta)>\epsilon,
\quad J_{\rm guard}(\theta')\ge J_{\rm guard}(\theta)-\delta.
\]

不满足则恢复全部 R。默认 epsilon=1e-5、delta=0。这里“无退化”仅指本次有限集合上的所选指标。反复使用 guard 仍然可能过拟合；guard 全部来自数学题，也不能保护编程、语言或其他能力。研究长期遗忘需要扩大保留集、多任务评测和独立的最终测试。

小于数值波动的提升没有可靠含义，应对 candidate 标签做步长敏感性、重复计算和精度检查；默认 epsilon 只是起点，不能当作统计显著性阈值。

## 3. Jev 接口和问题设计审查

官方 HTTP 为 `POST https://api.typesafe.ai/v1/systemone`，传 `model/state/questions`。本版直接使用 HTTP，避免 SDK 对象字段变化。`Score.probabilities` 使用字符串等级键，Choice 使用类别键，Noul 使用 `noul` 标量；Choice/Score 才有独立 confidence。详见 [TypeSafe API](https://docs.typesafe.ai/api) 和 [Score 文档](https://docs.typesafe.ai/primitives/score)。

每个问题给同一条轨迹做一个独立判断。confidence 反映返回分布的集中程度，不能当作“这次参数更新一定改善”的概率。[官方 confidence 说明](https://docs.typesafe.ai/confidence)也区分了模型确信程度与答案是否正确。

本版为以下问题建立固定槽位：

- 成功估计、格式遵循、循环论证、算术错误、无依据结论、冗长、遗漏、矛盾、计划相关性、结论一致性。
- reasoning 和 completion 的五级完整分布、归一化 score 与 confidence。
- 失败类型完整分布，包括 instruction、none、unknown。
- 前 12 个可见文本步骤的有效性、循环性、相关性、失败风险、依据充分性。
- 8 个实际生成 token 的抽样位置及上下文相关性。
- 工具和 memory 槽位；GSM8K 无这些事件时不发送问题，特征掩码为 0。
- 已观察步骤有效性估计的平均、最低与最高值。

这里步骤按非空文本行划分，是可见解答单元，不是内部推理状态。token 只抽样，未逐 token 全量评估。平均/最低/最高只覆盖保留的前 12 个步骤。问题数量可以调整，但需要新 schema、新干预记录和新 Router。

“第 n 步是否为所有可能策略中的最优步骤”一般没有可观测真值；“memory 有效性”在没有 memory 日志时也没有意义。本版不伪造这些值。现有特征仍是噪声代理，必须在任务数据上校验，问题更多并不自动更好。官方建议每个问题限定为简单、具体判断，参见 [TypeSafe Introduction](https://docs.typesafe.ai/introduction)。

## 4. Router 能学到什么

目标按每个 episode 单独构造：

\[
s_i=\Delta J_i\mathbf1[\Delta J_i>\epsilon],\quad
\alpha_i^*=s_i/\max(\max_j s_j,10^{-12}).
\]

全负或全零时 target 为全零。网络输出 sigmoid；使用 Smooth L1、成对 ranking loss 和小幅稀疏正则。忽略收益差小于 floor 的排序对。所有对都并列时 loss 仍保留可微的零值。

部署先按 alpha 排序，再要求超过阈值，最多选 Top-k，因此可以完全不更新。alpha 是相对效用分数，不是经过校准的成功概率。训练集内特征标准化参数随 Router 保存，验证集不参与标准化估计。

必须警惕：

1. SVD 按奇异值切块，可能只有少数方向总是有效，Router 只学会固定偏好。
2. Jev 看到的是文本行为，缺少梯度、模型状态等信息；相似轨迹的最佳方向不一定相同。
3. 单方向的最佳收益不等于多个方向联合更新的最佳收益。即使权重方向正交，模型输出和损失仍然非线性耦合。
4. Router 从固定模型状态收集的数据训练，在线累计更新后会发生分布漂移。代码允许从新的 adapter 锚点重新收集，但不会把一个旧 Router 自动解释成对所有后续权重都有效。
5. 默认 256 个 episode 只是管线与可学习性起点，270 维特征加隐藏层易过拟合。最终研究需更多数据、多种随机种子与消融。

本版给出 frozen base、random Top-k、all K、Router 四组同数据顺序对照。random 默认总选 k 个，Router 可以少选或弃权，两者更新预算不完全一致；all K 参数预算更多。应同时报告接受次数、选中数、训练步骤、总耗时和 API 开销，并进一步补充匹配更新预算的随机策略、固定方向、无 Jev 特征等消融，不能只看单个 Router 的最终分数。

## 5. 4090 与软件版本取舍

Qwen2.5 官方提供 1.5B 规格，本版用它实现“约 1B 级”实验。模型有 28 层，hidden size=1536；默认第 15 层（零基索引 14）的 q_proj，因此 K×r=64 小于 1536。参见 [模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 和 [官方配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)。

依赖固定为 Transformers 4.44.2 和 NumPy 1.26.4，保留用户已有 torch 2.1.2+cu118；该 torch/CUDA 组合由 [PyTorch 历史版本说明](https://pytorch.org/get-started/previous-versions/#v212)提供。没有使用要求较新 PyTorch 的当前版 Transformers，也没有用 Qwen3 强迫升级。

24GB 的配置策略是 BF16 冻结主干、FP32 增量、microbatch=1、长度 1024、非重入 checkpoint、128 个 completion token 一块的输出层损失。主干参数存储约数 GB，但实际峰值还取决于激活、生成缓存和 SVD 内存；没有在 AutoDL 4090 实测前，不给出保证的显存或耗时数字。

FP32 的 R 可以保留小更新，最终投影输出仍按主干 dtype 返回，BF16 舍入仍可能掩盖非常小的收益。需要按实际标签分布调整 lr/步数/目标层。默认 lr=0.01 比附件的 1e-4 大，是为极低维增量提供可测信号的实验起点，**不是验证过的最佳超参数**。
