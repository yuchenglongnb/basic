# 医疗大模型项目面试八股文
> 基于 HealthAI-2025 + MedicalGPT，涵盖数据构造 / 训练 / 评测全链路

---

## 一、项目整体定位（开场白）

> **"本项目以 HealthAI-2025 天池竞赛的临床病例数据为基础，结合 MedicalGPT 框架，搭建了一套完整的医疗大模型训练与评测闭环：用向量召回驱动数据构造、走完 SFT → Reward Model → PPO 的完整 RLHF 链路，并用 CEval 医疗指标 + PPL 双维度量化模型的真实提升。核心价值不在于跑通了一个框架，而在于数据、训练、评测三块互相咬合，形成可复现的改进闭环。"**

| 维度 | 内容 |
|---|---|
| 竞赛/数据来源 | HealthAI-2025 天池健康智能挑战赛（临床病例数据） |
| 训练框架 | shibing624/MedicalGPT（⭐4.6k） |
| 评测框架 | EleutherAI/lm-evaluation-harness |
| Base Model | Llama3.2-3B / Qwen2.5-3B |
| 训练阶段 | SFT → Reward Modeling → PPO（RLHF） |
| 核心改进 | 向量召回数据筛选 + 偏好数据合成 + CEval 医学指标提升 |

---

## 二、数据模块（重点！考察你对业务的理解）

### 2.1 数据来源概述

项目使用了两类数据源：

**① HealthAI-2025 竞赛原始病例数据**
- 包含真实/模拟病例：性别、年龄、主诉、现病史、体格检查、检查检验、诊断、处置建议等字段
- 是结构化的临床电子病历（EHR）格式，天然具有多字段对齐的特点
- 基于病人病情和病历特征，可用大模型合成推理链数据（Chain-of-Thought）

**② `shibing624/medical` 开源医疗数据集（240万条）**
- 预训练语料（PT）：大规模医学文本，用于增量预训练
- 指令微调数据（SFT）：11万条中英文医疗问诊对话
- 奖励模型数据（RM）：4000条偏好对，chosen = 医生答复，rejected = 本草模型答复

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

**Q：用什么 Embedding 模型？为什么？**

> A：优选 `BAAI/bge-large-zh-v1.5` 或 `text2vec-large-chinese`，两者在中文语义检索 MTEB 榜单上表现最优，且对医疗术语的语义理解优于通用 OpenAI Embedding。向量维度 1024，支持批量推理，效率可接受。

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
    --train_file_dir data/sft \
    --output_dir outputs/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --max_seq_length 2048
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

**RM 训练数据**：
```json
{
  "instruction": "高血压患者饮食注意事项？",
  "input": "",
  "output": [
    "低盐饮食，每日钠摄入<6g，多吃蔬菜水果...",  // chosen
    "随便吃，没有什么限制。"                        // rejected
  ]
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

MedicalGPT v2.4 新增，纯 RL 方法，类似 DeepSeek-R1 的训练思路：
- 无需 Critic 模型（省去一半显存）
- 用组内相对奖励替代绝对价值估计
- 可体验「aha moment」——模型自发产生推理链
- 脚本：`grpo_training.py` / `run_grpo.sh`

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

#### 指标 1：CEval physician（医学资格）
- **任务形式**：4选1单选题，考察医学知识掌握程度
- **评测方式**：0-shot 或 5-shot（few-shot）accuracy
- **基准水平**：随机猜测 25%，未微调 Llama3.2-3B 约 35~40%

```
SFT 后：CEval physician accuracy: xx% → xx%（↑N 个百分点）
```

#### 指标 2：PPL（Perplexity，困惑度）
- **定义**：$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$
- **含义**：模型对目标语料的建模能力，**越低越好**
- **评测语料**：在 1k 医疗领域测试文本上计算 PPL

```
训练前 PPL：xxx
SFT 后 PPL：xxx（↓改善 xx%）
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

> A：资源约束下的权衡。3B 模型用 LoRA 单卡（A100 40G）可完成 SFT，PPO 阶段 4 模型也可用 2 卡完成。更重要的是，医疗问答任务对模型的知识深度要求高但对通用推理能力要求相对稳定，3B 模型经过良好的 SFT+RLHF 后在 CEval 医学子集上的表现并不明显弱于 7B——边际收益递减，而 7B 训练成本翻了约 2~3 倍。

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
