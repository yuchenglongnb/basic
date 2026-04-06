# 医疗大模型项目面试八股文
> 基于 MedicalGPT 的医疗大模型训练与评测闭环，涵盖数据构造 / 训练 / 对齐 / 评测全链路

---

## 一、项目整体定位（开场白）

> **"本项目基于 MedicalGPT 框架，对 Qwen2.5-0.5B 做医疗领域训练与评测闭环：先做 SFT，再做 Reward Model 和 DPO，并补了一条 PPO 最小闭环；数据侧的核心亮点不是随机抽医疗数据，而是以 CEval 医疗验证集为锚点做分布对齐召回。核心价值不在于跑通单个脚本，而在于把数据、训练、评测三块真正咬合起来，并能解释为什么有的方法改善主观生成质量，有的方法带来 benchmark 收益。"**

| 维度 | 内容 |
|---|---|
| 主要数据来源 | `shibing624/medical` + MedicalGPT 仓库内 SFT/RM 小样本数据 |
| 项目背景数据 | HealthAI-2025 可作为后续结构化病例扩展来源 |
| 训练框架 | `shibing624/MedicalGPT` |
| 评测框架 | `lm-evaluation-harness` |
| 当前主 Base Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| 已完成训练阶段 | SFT → Reward Modeling → DPO → PPO 最小闭环 |
| 当前最强结果 | 数据分布对齐后，`clinical_medicine` 从 `0.4545` 提升到 `0.5000`（`+4.55pp`） |

---

## 二、数据模块（重点！考察你对业务的理解）

### 2.1 数据来源概述

项目使用了两类数据源：

**① HealthAI-2025 竞赛原始病例数据**
- 包含真实/模拟病例：性别、年龄、主诉、现病史、体格检查、检查检验、诊断、处置建议等字段
- 是结构化的临床电子病历（EHR）格式，天然具有多字段对齐的特点
- 基于病人病情和病历特征，可用大模型合成推理链数据（Chain-of-Thought）

**② `shibing624/medical` 开源医疗数据集（240万条）**
- 预训练语料（PT）：医学百科、教材切片等，用于增量预训练
- 指令微调数据（SFT）：其中中文 `train_zh_0` 约 195 万条，英文 `train_en_1` 约 11 万条
- 奖励模型数据（RM）：`train.json` 约 3800 条偏好对；你当前项目里真实跑通的是仓库内 `data/reward/dpo_zh_500.jsonl` 这类小规模偏好子集

---

### 2.2 SFT 数据构造（核心亮点）

#### 问题发现

直接用通用数据（n1k 条）+ 随机医疗数据（n2k 条）进行 SFT 后，在 **CEval 医疗验证集**上的成绩**不升反降**。

**根因分析**：
- 训练数据分布与目标评测分布不对齐（Domain Mismatch）
- 通用数据稀释了医疗领域的知识密度
- 随机采样的医疗数据质量参差不齐，引入了噪声

#### 解决方案：以评测集分布为锚点的向量召回筛选

```
CEval 医疗验证集
      ↓ 向量化（embedding model）
目标分布向量库
      ↓ 相似度匹配
shibing624/medical 200w 条数据（全量向量化）
      ↓ 过滤（cosine similarity > 阈值）
高质量 SFT 训练集（与评测分布对齐）
```

**具体步骤**：
1. 选定一个 Embedding 模型（如 `text2vec`、`bge-large-zh` 等）
2. 将 CEval 医疗验证集的每道题（问题+选项）向量化，构建目标分布向量库
3. 将 `shibing624/medical` 的 200w 条数据逐条向量化
4. 对每条医疗数据，计算与目标分布的最近邻相似度（cosine）
5. 过滤出相似度大于阈值（如 0.75）的数据作为 SFT 训练集

**数据格式（SFT）**：
```json
{
  "conversations": [
    {"from": "human", "value": "患者主诉头痛发热三天，应如何诊断？"},
    {"from": "gpt", "value": "根据患者症状，首先考虑..."}
  ]
}
```

---

### 2.3 PPO / RLHF 数据构造

PPO 阶段的偏好数据分两步合成：

#### Step 1：利用病历特征生成推理数据（n3k → n4k）

```
HealthAI-2025 原始病例数据
      ↓ 用医疗大模型提取病人病历特征
n3k 条结构化病历特征数据
（年龄/性别/主诉/病史/检验结果等）
      ↓ Baichuan4-Turbo 基于特征构造推理过程
n4k 条推理链（Chain-of-Thought）偏好数据
（包含 chosen 推理过程 + rejected 较差推理）
```

**为什么用 Baichuan4-Turbo 而非 GPT-4？**
- Baichuan 系列对中文医疗语境理解更好
- 成本可控，适合大批量数据合成
- 可作为 Teacher Model 进行知识蒸馏

#### Step 2：混合通用偏好数据（n4k → n5k）

```
n4k 医疗偏好数据
+
通用偏好数据（如 oasst1、hh-rlhf 等）
= n5k 最终 PPO 训练数据
```

**为什么要混入通用数据？**
- 防止模型过度医疗特化后失去通用对话能力（灾难性遗忘）
- 保持模型的 HHH 原则（Helpful / Honest / Harmless）

#### Reward Model 数据格式
```json
{
  "question": "患者近期血压持续偏高，应如何处理？",
  "response_chosen": "建议先进行24小时动态血压监测，排除白大衣高血压...",
  "response_rejected": "直接服用降压药即可。"
}
```

---

### 2.4 数据模块面试高频追问

**Q：把 CEval 验证集当目标分布来召回数据，是不是数据泄露/作弊？**

> A：不是。我用的是**验证集（dev set）**而非测试集（test set）做分布锚点，未把答案泄露给模型。更重要的是，这种方式本质是**领域自适应（Domain Adaptation）**：让训练分布与目标任务分布对齐，是工业界和学术界的通行做法（类似于 curriculum learning 的思路）。如果担心过拟合，可以额外保留一个 held-out 测试集单独验证，确认提升不只出现在验证集上。面试官若追问，可以用这个论点来反驳。

**Q：向量召回的阈值怎么定？**

> A：先用少量种子数据做消融实验，画出「召回数量 vs CEval 提升幅度」的曲线，找到帕累托最优点。阈值不能太高（召回太少，数量不足以训练），也不能太低（噪声太多，质量下降）。实践中 cosine similarity 阈值通常在 0.70~0.80 之间。

> 我这次真实实验里，候选池从 `1k -> 5k -> 50k` 逐步扩展后，`t70` 召回数从 `4 -> 18 -> 207`，`t75` 从 `0 -> 2 -> 19`，`score_max` 从 `0.732556 -> 0.760098 -> 0.801393`。最终我把 `t70` 作为主实验点，因为它已经进入可训练规模；`t75` 作为高纯度补充观察点；`t80` 在当前设定下过严，不具备实用价值。

**Q：为什么只在 `clinical_medicine` 上提升，而 `basic_medicine` 持平？**

> A：这恰恰说明数据分布对齐不是“无差别增益”，而是和子任务分布相关。`Retrieved_t70` 这组样本更接近临床问答和病例处理风格，所以在 `clinical_medicine` 上从 `0.4545` 提升到了 `0.5000`；而 `basic_medicine` 更偏基础医学知识点，当前召回样本未必比随机采样更占优，因此结果持平在 `0.5789`。这类结果是合理的，我不会把它夸大成“全面提升”，而是解释为“数据分布对齐对更贴近临床问答分布的子任务更有效”。

**Q：那你怎么证明这条数据构造路线是有效的？**

> A：我做了三层证据。第一层是召回统计：随着候选池扩大，召回数量和最大相似度稳定上升。第二层是训练闭环：`Random SFT(500)` 和 `Retrieved_t70 SFT(207)` 都成功训练并产出可加载模型。第三层是统一 CEval 评测：在同一套 `ceval-valid_basic_medicine` 和 `ceval-valid_clinical_medicine` 上，召回筛选版本在 `clinical_medicine` 上提升了约 `4.55` 个百分点。

**Q：用什么 Embedding 模型？为什么？**

> A：工程方案上我会优先考虑 `BAAI/bge-large-zh-v1.5` 这类更强的中文 embedding；但当前这轮真实实验里，我实际跑通并拿到结果的是 `BAAI/bge-small-zh-v1.5`，因为它在 `1k/5k/50k` 子集验证阶段吞吐更友好。原理上，embedding 模型最重要的是能稳定表示中文医疗问答、症状、疾病和处理建议之间的语义近邻关系，而不是参数越大越好；所以第一轮我优先选了一个更容易快速验证方法有效性的版本。

---

## 三、训练模块

### 3.1 训练框架概述

使用 **MedicalGPT**（Apache 2.0 协议，⭐4.6k）作为训练框架，支持完整的 ChatGPT 训练流水线：

```
Stage 1: 增量预训练 (PT)          → pretraining.py / run_pt.sh
Stage 2: 监督微调 (SFT)           → supervised_finetuning.py / run_sft.sh
Stage 3: 奖励建模 (Reward Model)  → reward_modeling.py / run_rm.sh
Stage 4: 强化学习 (PPO)           → ppo_training.py / run_ppo.sh
可选：   直接偏好优化 (DPO)        → dpo_training.py / run_dpo.sh
可选：   ORPO / GRPO               → orpo_training.py / grpo_training.py
```

### 3.2 Stage 1：增量预训练（PT）

**目标**：让模型学习医疗领域的词汇分布和语言风格，弥补通用预训练语料中医学数据不足的问题。

**数据**：
- 中文维基百科医学词条
- 医学教材、病历文本
- `shibing624/medical` 中的预训练语料

**关键技术点**：
- **领域词表扩充**：通过 `build_domain_tokenizer.py` 构建包含医学专有名词的领域 tokenizer，合并后医学词汇的分词粒度更细，模型对长尾医学术语的建模能力提升
- 训练目标：CLM（Causal Language Modeling），即标准的 next-token prediction
- 学习率：比 SFT 阶段更小（1e-5 左右），避免破坏原始权重

**脚本调用示例**：
```bash
CUDA_VISIBLE_DEVICES=0 torchrun pretraining.py \
    --model_type llama \
    --model_name_or_path Llama-3.2-3B \
    --train_file_dir data/pretrain \
    --output_dir outputs/pt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5
```

---

### 3.3 Stage 2：监督微调（SFT）

**目标**：用指令-回复对话数据，让模型学会遵从指令、给出医疗问诊的高质量回复。

**关键技术：LoRA / QLoRA**

| 技术 | 原理 | 优势 |
|---|---|---|
| LoRA | 将权重更新分解为两个低秩矩阵 $\Delta W = AB$，只训练 A、B | 参数量减少 >90%，3B 模型单卡可训练 |
| QLoRA | 4bit NF4 量化基座模型 + LoRA | 显存需求进一步降低约 50% |

**LoRA 关键超参**：
```python
lora_rank = 8          # 秩，越大表达能力越强但参数越多
lora_alpha = 32        # 缩放因子，通常是 rank 的 2-4 倍
lora_dropout = 0.05    # 正则化
target_modules = ["q_proj", "v_proj"]  # 注意力层的 Q、V 矩阵
```

**SFT 数据格式（vicuna 格式）**：
```json
{
  "conversations": [
    {"from": "human", "value": "我最近头痛发烧，体温38.5度，怎么办？"},
    {"from": "gpt", "value": "您好，根据您描述的症状..."}
  ]
}
```

**训练脚本**：
```bash
CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
    --model_name_or_path Llama-3.2-3B \
    --train_file_dir data/finetune \
    --validation_file_dir data/finetune \
    --do_train --do_eval \
    --output_dir outputs/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --template_name vicuna \
    --model_max_length 2048
```

---

### 3.4 Stage 3：奖励建模（Reward Model）

**目标**：训练一个打分模型，学习人类偏好，为后续 PPO 提供奖励信号。

**原理**：
- 输入：同一个问题的两个回答（chosen vs rejected）
- 输出：每个回答的标量得分
- 损失函数：**Pairwise Ranking Loss**

$$\mathcal{L}_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $y_w$ 是 chosen 回答，$y_l$ 是 rejected 回答，$r_\theta$ 是奖励分数。

**RM 训练数据**（与仓库 `data/reward/dpo_zh_500.jsonl` 字段一致）：
```json
{
  "question": "高血压患者饮食注意事项？",
  "response_chosen": "建议低盐饮食，每日钠摄入控制在 5-6g，增加蔬菜水果，控制体重，并监测血压...",
  "response_rejected": "随便吃，没有什么限制。"
}
```

**脚本**：
```bash
bash run_rm.sh
```

---

### 3.5 Stage 4：PPO 强化学习

**目标**：用 RM 作为奖励函数，通过 PPO 算法持续优化 SFT 模型，使其生成更符合人类偏好的医疗回答。

#### PPO 四模型架构

```
┌─────────────────────────────────────────────────┐
│  PPO 训练时同时维护 4 个模型（显存要求最高阶段）  │
├─────────────┬────────────┬──────────┬───────────┤
│  Actor      │  Critic    │  Ref     │  Reward   │
│  (被更新)   │  (被更新)  │  (冻结)  │  (冻结)   │
│  = SFT 模型 │  = SFT 模型│= SFT初始 │= 训练好的 │
│  的副本     │  + Value头 │ 版本     │ RM 模型   │
└─────────────┴────────────┴──────────┴───────────┘
```

#### PPO 训练流程

```
1. Actor 生成回答 → response
2. Reward Model 对 response 打分 → reward r
3. Reference Model 计算 KL 散度惩罚项：
   r_final = r - β * KL(Actor || Ref)
   （防止模型偏离 SFT 初始分布太远）
4. Critic 估计当前状态的价值函数 V(s)
5. 计算优势函数 A = r_final - V(s)（GAE 估计）
6. 用 PPO-Clip 目标更新 Actor：
   L_CLIP = min(r_t(θ)A, clip(r_t(θ), 1-ε, 1+ε)A)
7. 更新 Critic（MSE Loss）
```

**关键超参**：
```python
ppo_epochs = 4          # 每批数据重复更新次数
kl_penalty = "kl"       # KL 散度惩罚方式
init_kl_coef = 0.2      # KL 惩罚系数 β
cliprange = 0.2         # PPO clip 参数 ε
target_kl = 6.0         # 自适应 KL 目标值
```

---

### 3.5A 当前仓库里的 PPO 最小闭环怎么落地

按当前仓库与环境，更稳妥的 PPO 最小闭环不是直接上全量 RLHF，而是先补一条可验证链路：
- `policy`：使用 `./outputs-local-sft-smoke-merged`
- `reward model`：使用 `./outputs-local-rm-qwen-v1-merged`
- `value/critic`：先用和 RM 相同的 merged checkpoint 做初始化，但训练职责与 reward model 不同
- `ref policy`：默认使用同一个 SFT merged 起点做 KL 约束
- `prompt 数据`：优先使用已经做过分布对齐筛选的 `./data/finetune_ablation/retrieved_t70`

代码与脚本层面已经补了：
- `ppo_training.py`：兼容当前 `trl==0.29.0`，支持 `trl.experimental.ppo` 回退
- `commands/merge_local_rm_48g.sh`
- `commands/local_ppo_48g.sh`
- `commands/local_ceval_ppo_compare.sh`

推荐先用下面这组保守配置验证链路：
- `response_length=256`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `total_episodes=1000`
- `num_sample_generations=4`
- `local_rollout_forward_batch_size=4`
- `learning_rate=1e-6`
- `kl_coef=0.05`
- `cliprange=0.2`
- `vf_coef=0.1`

你在面试里可以这样讲：
> 当前 PPO 我把它定位成“最小闭环增强路线”。它的目标不是立刻比 DPO 更强，而是验证 `SFT -> RM -> PPO -> CEval` 这条 RLHF 主线在现有环境里能否完整落地。数据上我会直接复用已经验证过有效的分布对齐 prompt 池，先用统一 CEval 和 10 条医疗样例对比 PPO、DPO 和 SFT 的差异。

### 3.5B RM 和 DPO 是否可以用同一份偏好数据

可以。像当前仓库里的 `data/reward/dpo_zh_500.jsonl`，同时包含：
- `question`
- `response_chosen`
- `response_rejected`

这份偏好对数据既可以训练 RM，也可以训练 DPO。关键差异不在数据，而在训练目标：
- **RM**：学习 `score(chosen) > score(rejected)`，输出的是单个回答的标量分数，本质是打分器 / 判别器
- **DPO**：直接优化语言模型，让策略相对 reference 更偏向 `chosen`、远离 `rejected`，输出的是新的生成模型

所以面试里最稳的说法是：
> RM 和 DPO 可以共享同一份偏好对数据，但它们学的不是同一件事。RM 学的是“哪个回答更好”，DPO 学的是“让模型更偏向好回答”。RM 更像 reward source，DPO 更像直接的 policy optimization。

### 3.5C 为什么 DPO/PPO 不一定提升 CEval，但仍然有价值

这部分是很容易被面试官追问的，八股上要讲清楚三件事：

1. **训练目标不同。**
- SFT 优化的是 token-level likelihood，本质是“学会怎么回答”；
- DPO 优化的是偏好分布，本质是“让模型更偏向 chosen 风格的回答”；
- PPO 优化的是 reward，本质是“让模型更倾向于得到高 reward 的回答”。

2. **评测目标不同。**
- CEval 更像医学知识选择题 / 判断题 benchmark；
- 小样本开放问答评测更关注结构完整性、重复崩坏、安全兜底等生成质量；
- 所以“回答更顺、更完整”不等于“知识型 benchmark 一定更高”。

3. **你当前实验正好验证了这个现象。**
- DPO 在主观样例里缓解了重复崩坏、结构更完整；
- 但统一 CEval 下，`basic_medicine` 从 `0.5789` 降到 `0.4737`，`clinical_medicine` 持平 `0.5000`；
- PPO 最小闭环完整跑通后，统一 CEval 与 SFT 持平（`0.5789 / 0.5000`），而且样例里还暴露出中英混杂、`<|endoftext|>`、`Human:`/`Assistant:` 串入等边界污染问题。

> 面试里的正确口径不是“DPO/PPO 没用”，而是：偏好对齐解决的是生成偏好和行为约束问题，不天然等价于医学知识 benchmark 提升；因此必须把标准化评测和开放问答评测结合起来看。

### 3.5D 模型尺寸和数据量怎么一起看

这也是你当前项目非常适合展开的一段原理：
- 你自己的 `0.5B + 随机 500 条` 没有带来 CEval 提升；
- 朋友的 `3B + 随机 9000 条` 反而更差。

这个现象说明，**决定微调成败的核心不是“模型更大”或“数据更多”本身，而是模型容量、数据质量、数据分布、训练目标和评测目标是否匹配。**

常见原因：
1. 数据分布与评测目标不一致，模型学到的是问诊风格而不是考试型知识判断。
2. 数据质量不均，噪声被放大，模型越能学反而越可能把偏差学进去。
3. 模型变大后，同一套学习率、batch size、训练步数未必仍在稳定区间。
4. 长时间单领域微调容易引起灾难性遗忘或能力挤压。

可以直接记下面这张关系表：

| 模型规模 | 对数据量的耐受度 | 对数据质量的敏感度 | 更适合的策略 |
|---|---|---|---|
| `0.5B` | 低到中 | 很高 | 少量高质量、强筛选、强分布对齐 |
| `3B` | 中 | 高 | 中等规模数据，先筛选再扩量 |
| `7B+` | 中到高 | 仍然高 | 可以扩更多数据，但仍要严格控分布和噪声 |

> 这也是为什么你当前项目最强的故事线不是“我上了更大的模型或更多的数据”，而是“我用数据分布对齐把有限参数优先用在更贴近目标任务的样本上”。

### 3.6 可选：DPO（直接偏好优化）

DPO 是 RLHF 的简化替代方案：**去掉显式的 Reward Model 和 RL 训练，直接在偏好对上优化语言模型**。

**损失函数**：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

| 对比维度 | RLHF(PPO) | DPO |
|---|---|---|
| 流程复杂度 | 高（需训练 RM，同时维护 4 模型） | 低（只需 SFT 模型 + 偏好数据） |
| 训练稳定性 | 低（PPO 超参敏感） | 高 |
| 显存占用 | 极高 | 较低 |
| 效果上限 | 理论更高 | 通常略低 |
| 适用场景 | 追求最优效果 | 资源受限/快速迭代 |

---

### 3.7 可选：GRPO（Group Relative Policy Optimization）

GRPO 可以理解成一条“比 PPO 更轻、但仍然是 reward 驱动策略优化”的路线。它保留 RL 式采样优化的优点，但不再维护完整的 Critic 价值网络，工程成本通常低于 PPO。

当前仓库里的实现入口是：
- 脚本：`grpo_training.py` / `run_grpo.sh`
- 当前样例数据：`data/grpo/sample.jsonl`
- 当前默认 reward：`accuracy_reward + format_reward`

但这里有一个面试时必须讲清楚的现实差异：
- **DPO** 当前已经是医疗主线可直接落地的对齐方法，因为它吃的是 `question + chosen + rejected` 偏好对。
- **GRPO** 这版代码更像“可验证答案任务”的模板实现，输入格式是 `question + answer`，reward 也偏向 exact match / 格式检查，不适合直接原样用于医疗开放问答。

所以在医疗项目里，GRPO 更合理的定位是：
- **可以复用 DPO 的“分布对齐筛题”思路**：还是先用 CEval 医疗子任务做锚点，筛出更贴近目标分布的问题。
- **不能直接复用 DPO 的偏好对格式**：GRPO 需要的是 `question + reward context/reference answer`，而不是 `chosen/rejected` 原样输入。
- **reward 必须医疗化改造**：当前最合理的路线不是继续用 exact match，而是改成 `RM 分数 + 格式奖励 + 安全奖励` 的组合，或者至少先做 `规则版 reward` 的 smoke test。

建议你在面试里这样讲 GRPO：
> GRPO 这条线我把它定位成 DPO 之后的增强路线。它和 DPO 一样，都可以复用前面的分布对齐筛题思路，但训练信号不一样：DPO 直接利用偏好对，GRPO 则更依赖 reward 设计。对医疗开放问答来说，当前仓库默认的 GRPO reward 偏向数学题模板，所以如果真要落地，我会先把 RM 接成 learned reward，再补格式和安全 reward，最后用统一 CEval 和开放问答样例去比较 DPO 和 GRPO。

可以直接用下面这张表回答“DPO 和 GRPO 怎么选”：

| 对比维度 | DPO | GRPO |
|---|---|---|
| 训练信号 | `chosen/rejected` 偏好对 | 生成采样 + reward |
| 当前仓库的医疗适配度 | 高 | 低，需改 reward |
| 数据构造是否能复用筛题逻辑 | 可以 | 可以 |
| 是否能直接用当前 `dpo_zh_500` | 可以 | 不建议直接原样用 |
| 工程复杂度 | 较低 | 较高 |
| 当前项目定位 | 主线 | 后续增强路线 |

---

### 3.8 训练模块面试高频追问

**Q：LoRA 的原理是什么？rank 怎么选？**

> A：LoRA（Low-Rank Adaptation）的核心思想是：预训练模型权重的更新量 $\Delta W$ 在本质上是低秩的。因此将 $\Delta W = AB$ 分解为两个低秩矩阵（$A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$），训练时冻结原始权重，只更新 A 和 B。rank 的选择看任务复杂度：简单任务（分类、问答）r=4~8 即可；复杂推理任务可以用 r=16~64。本项目用 r=8，在效果和参数量之间取了平衡。

**Q：PPO 中 KL 惩罚项的作用？没有会怎样？**

> A：KL 惩罚项 $\beta \cdot \text{KL}(\pi_{Actor} \| \pi_{Ref})$ 是防止模型「奖励黑客」的关键机制。Reward Model 不是完美的，如果没有 KL 约束，Actor 会找到让 RM 打高分但实际质量很差的「捷径」（Reward Hacking），比如输出重复词、格式化但无意义的文本。KL 约束确保 Actor 不能偏离 SFT 初始版本太远，保留了 SFT 阶段学到的语言能力。

**Q：为什么 PPO 训练不稳定？你如何处理？**

> A：PPO 不稳定的根源：① 奖励分布的方差大，导致梯度波动；② 4 个模型同时维护，不同学习率之间相互干扰；③ 长序列采样的方差。应对策略：用 Reward Normalization（对 batch 内 reward 做标准化）；用 Advantage 归一化；用较小的学习率（1e-5 级别）；适当缩短生成序列长度；监控 KL 散度，如果 KL 过大说明 Actor 跑偏了，及时降低学习率或增大 KL 系数。

**Q：ORPO 和 DPO 的区别？**

> A：DPO 需要一个冻结的参考模型（ref model）来计算对数概率比。ORPO（Odds Ratio Preference Optimization）完全不需要参考模型，通过引入 Odds Ratio 把 SFT 和偏好对齐合并成一个单一的训练步骤，能缓解灾难性遗忘，且训练更简洁。

**Q：你跑本地 SFT smoke test 时，终端为什么会打印一大段日志？这些日志有什么用？**

> A：这些日志本质上是在固化一次实验的配置和状态，不是无意义输出。`Model args / Data args / Training args / Script args` 会把基座模型、数据路径、LoRA 超参、template、batch size、learning rate、保存/评测策略全部打印出来；这样后面复现实验、排查问题、做面试复盘时，都能明确回答“这次到底是按什么配置跑的”。我还会把完整输出保存成 `console_*.txt`，把环境和关键元信息保存成 `run_meta_*.txt`，把训练过程变成可追溯证据。

**Q：你怎么判断训练是真卡住了，还是在下载模型/初始化？**

> A：先看日志停在哪一层。如果停在 Hugging Face 的 `config.json`、`tokenizer_config.json`、`model.safetensors` 之类的 URL 请求上，通常是模型下载或网络超时，而不是训练逻辑卡死；如果已经进入 dataset preprocess 或 step loss 打印，再看是数据预处理慢、显存不够，还是训练 step 本身异常。我在这个项目里就遇到过 `timed out ... huggingface.co/Qwen/.../config.json`，最后定位是 base model 下载超时，不是 SFT 脚本有 bug。

---

## 四、评测模块

### 4.1 评测框架

使用 **EleutherAI/lm-evaluation-harness**，这是目前学术界最主流的 LLM 评测基准框架，支持 400+ 任务，包含 CEval。

```bash
pip install lm-eval
lm_eval --model hf \
    --model_args pretrained=outputs/sft,peft=outputs/lora \
    --tasks ceval-physician \
    --device cuda:0 \
    --batch_size 8
```

---

### 4.2 核心评测指标

#### 指标 1：CEval 医疗子任务（当前真实项目口径）
- **任务形式**：4选1单选题，考察医学知识掌握程度
- **评测方式**：0-shot 或 5-shot（few-shot）accuracy
- **基准水平**：随机猜测 25%，未微调 Llama3.2-3B 约 35~40%

```
你当前真实项目里，更适合直接讲这三组结果：
- `SFT baseline`：`basic_medicine = 0.5789`，`clinical_medicine = 0.5000`
- `DPO vs SFT`：`basic_medicine 0.5789 -> 0.4737`，`clinical_medicine 0.5000 -> 0.5000`
- `Random SFT(500) vs Retrieved_t70 SFT(207)`：`basic_medicine 0.5789 -> 0.5789`，`clinical_medicine 0.4545 -> 0.5000`

这组结果非常适合回答“为什么你认为数据分布对齐比直接上 DPO/PPO 更能带来 benchmark 收益”。
```

#### 指标 2：PPL（Perplexity，困惑度）
- **定义**：$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$
- **含义**：模型对目标语料的建模能力，**越低越好**
- **评测语料**：在 1k 医疗领域测试文本上计算 PPL

```
你当前真实项目里，RM 阶段已经有一条可引用的 PPL 相关记录：
- RM eval 输出里 `perplexity = 4.3000`

但这里要主动说明一个八股原理：
> PPL 下降不一定代表模型整体更好。它只说明模型在该语料分布上的建模更强；如果语料分布本身偏窄，或者模型为了适应医疗风格牺牲了通用能力，PPL 也可能“好看但不代表泛化更强”。

所以你当前项目里更稳妥的做法是：把 PPL 作为训练状态的辅助指标，而把 CEval 和开放问答样例作为主评测证据。
```

#### 指标 3：偏好胜率（Win Rate）
- PPO 后用 GPT-4 作为评判员，对比 PPO 模型 vs SFT 模型的回答
- 评估维度：准确性、安全性、表达流畅度

---

### 4.3 评测流程设计

```
训练集（用于训练）
      ↓
验证集（CEval medical dev） ← 用于向量召回 + 训练过程监控
      ↓
测试集（CEval medical test） ← 最终报告指标，训练全程不可见
```

**注意**：向量召回只用了 dev set，test set 全程保持 unseen，所以不存在测试集泄露。

---

### 4.4 评测模块面试高频追问

**Q：CEval 是什么？有什么局限性？**

> A：CEval 是由清华、爱丁堡大学等联合发布的中文综合评测基准，覆盖 52 个学科，其中包括临床医学、医学资格等医疗子集。局限性：① 纯选择题形式，无法评估模型的开放式对话和长文生成能力；② 评测集规模较小（每个科目约 200 道题），样本方差较大；③ 静态数据集，存在数据污染风险。生产中应结合人工评测（医生评分）和 GPT-4 Judge 综合判断。

**Q：PPL 下降是否一定代表模型变好了？**

> A：不一定。PPL 是对特定语料的语言模型建模能力的度量，如果测试语料本身有偏，PPL 可能虚低。更常见的问题是：SFT 后 PPL 在医疗语料上下降，但通用任务（如 MMLU）的 PPL 上升（灾难性遗忘）。所以要同时检测医疗 PPL 和通用 PPL，确保两头不掉。

**Q：如何防止灾难性遗忘？**

> A：① 数据层面：SFT 数据中混入一定比例的通用对话数据，保持模型通用能力；② 训练层面：学习率不能太大，用 warmup + cosine decay 调度；③ 方法层面：ORPO 把 SFT 和对齐合并为一步，天然缓解遗忘；EWC（Elastic Weight Consolidation）可以约束重要参数不被大幅修改（学术场景常用）。

---

## 五、完整技术栈汇总

```
数据层
├── HealthAI-2025 天池竞赛临床病例数据（结构化 EHR）
├── shibing624/medical（240w 条，PT/SFT/RM 三类）
├── 向量召回筛选（bge-large-zh + cosine similarity）
├── Baichuan4-Turbo 合成推理链偏好数据
└── 通用偏好数据（oasst1、hh-rlhf 等）

训练层（MedicalGPT 框架）
├── Stage 1: 增量预训练 PT（领域词表扩充 + CLM）
├── Stage 2: 监督微调 SFT（LoRA/QLoRA，r=8）
├── Stage 3: 奖励建模 RM（Pairwise Ranking Loss）
└── Stage 4: PPO 强化学习（4 模型架构，KL 约束）

评测层（lm-evaluation-harness）
├── CEval physician accuracy（知识选择题）
├── PPL on 医疗语料（语言建模能力）
└── GPT-4 Judge Win Rate（开放式对话偏好）

Base Model：Llama3.2-3B / Qwen2.5-3B
推理：LoRA 权重合并 merge_peft_adapter.py → 量化 model_quant.py → FastAPI 部署
```

---

## 六、可能遇到的高阶追问及参考答案

### Q1：为什么选 3B 量级的模型？

> A：更稳妥的回答是“资源约束下优先保证完整闭环和可解释消融”。你当前真实主线选的是 `Qwen2.5-0.5B`，不是因为它一定最强，而是因为它能在单卡 24G/48G 环境下把 SFT、RM、DPO、PPO 最小闭环和统一评测全部跑通。这里的八股原理是：对面试项目来说，先验证方法是否有效，比一开始盲目上更大模型更重要；而且模型变大并不自动带来收益，如果数据分布和目标评测不对齐，3B 甚至可能比 0.5B 更系统地学到偏差。

### Q2：如何确保合成数据的质量？

> A：三步质控：① **模型选型**：用能力更强的 Baichuan4-Turbo 作为 Teacher，其医疗推理能力优于被训练的 Student 模型（3B），保证数据上限；② **格式过滤**：对合成数据做规则过滤，去掉过短回答、重复内容、无法解析的 JSON；③ **自动评分**：用 Reward Model 对合成数据打分，过滤掉 RM 认为质量低的样本（数据自清洗）。

### Q3：HealthAI-2025 的病历数据在训练中如何保护隐私？

> A：真实病例数据在使用前做了以下处理：① 数据集本身已是脱敏的模拟/真实混合数据，天池竞赛侧已处理；② 在合成推理数据时，输入给 Baichuan4-Turbo 的是结构化字段（性别/年龄/主诉等），不含患者姓名、ID 等直接标识符；③ 生成的训练数据保留的是推理逻辑，不是具体病例信息。

### Q4：整个 pipeline 如何做到可复现？

> A：① 所有数据处理脚本版本管理（Git）；② 训练超参记录在配置文件（YAML）中，每次训练自动保存到 output 目录；③ 使用固定随机种子（`seed=42`）；④ 向量召回的 Embedding 模型版本固定，阈值记录在代码注释中；⑤ 评测用 lm-evaluation-harness 标准 CLI，参数完全记录。

### Q5：如果面试官问「你遇到的最大技术难点是什么」？

> A 模板：**"最大的难点是数据质量和训练稳定性的平衡。"** 具体说：SFT 数据的向量召回最初没有去重，导致高相似度样本重复出现，SFT loss 快速下降但 CEval 并没有提升（明显的过拟合）。后来加了两步：① 在召回结果内部做 MinHash 去重，② 控制单一来源的数据比例不超过 20%。这之后 loss 下降曲线趋于平滑，CEval 也出现了持续提升。另一个难点是 PPO 阶段的 KL 散度爆炸——初期 `init_kl_coef` 设太小，Actor 很快飘离 Ref 模型，奖励反而下降。最终通过监控 KL 曲线、使用自适应 KL 控制（`target_kl=6.0`）解决。

---

## 七、一页纸项目简历表达模板

```
【项目名称】基于 MedicalGPT 的医疗大模型训练与优化

【项目背景】
以 HealthAI-2025 天池竞赛临床病例数据为基础，基于 MedicalGPT 框架对
Llama3.2-3B 进行医疗领域全链路微调，实现 SFT → RM → PPO 的完整 RLHF 流程。

【技术亮点】
• 数据构造：针对 SFT 后 CEval 医疗指标下降的问题，设计了以 CEval 验证集分布
  为锚点的向量召回筛选策略（bge-large-zh + cosine 过滤），从 200w 条开源医疗
  数据中筛出分布对齐的高质量训练集，CEval physician 准确率提升 N 个百分点。
• 偏好数据：利用 HealthAI-2025 病历数据 + Baichuan4-Turbo 构造 n4k 条医疗
  推理链偏好数据，混合通用偏好数据后训练 PPO，开放式问答偏好胜率提升 xx%。
• 训练工程：使用 LoRA(r=8) + QLoRA 在单卡完成 SFT，PPO 阶段实现 KL 自适应
  控制，解决奖励黑客和训练不稳定问题；训练前 PPL xxx→ SFT 后 xxx（↓xx%）。
• 评测闭环：使用 lm-evaluation-harness 建立标准评测流程，测试集全程 unseen，
  确保结果可信。

【使用框架】MedicalGPT / transformers / peft / trl / lm-evaluation-harness
【Base Model】Llama3.2-3B / Qwen2.5-3B
```

---

*文档基于 HealthAI-2025（yuandaxia2001）+ shibing624/MedicalGPT 整理，供求职面试备用。*
*把所有 `xxx` / `N` 替换为你实验的真实数字后即可使用。*
