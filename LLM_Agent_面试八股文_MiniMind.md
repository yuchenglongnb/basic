# 大模型 & Agent 面试八股文
> 结合 MiniMind 源代码，整理面试高频考点与训练细节

---

## 目录

1. [Transformer 架构基础](#1-transformer-架构基础)
2. [RMSNorm](#2-rmsnorm)
3. [RoPE 旋转位置编码](#3-rope-旋转位置编码)
4. [GQA 分组查询注意力 & KV Cache](#4-gqa-分组查询注意力--kv-cache)
5. [Flash Attention](#5-flash-attention)
6. [SwiGLU 激活函数 & FFN](#6-swiglu-激活函数--ffn)
7. [MoE 混合专家模型](#7-moe-混合专家模型)
8. [预训练 Pretraining](#8-预训练-pretraining)
9. [SFT 监督微调](#9-sft-监督微调)
10. [LoRA 低秩适配](#10-lora-低秩适配)
11. [DPO 直接偏好优化](#11-dpo-直接偏好优化)
12. [GRPO / RLHF](#12-grpo--rlhf)
13. [知识蒸馏](#13-知识蒸馏)
14. [混合精度训练 & 梯度相关技巧](#14-混合精度训练--梯度相关技巧)
15. [分布式训练 DDP](#15-分布式训练-ddp)
16. [学习率调度](#16-学习率调度)
17. [Tokenizer](#17-tokenizer)
18. [Agent 相关](#18-agent-相关)

---

## 1. Transformer 架构基础

### 整体结构

MiniMind 采用 **Decoder-only** 架构（类 LLaMA），每个 `MiniMindBlock` 包含：

```
输入 → Pre-Norm(RMSNorm) → Self-Attention → 残差连接
     → Pre-Norm(RMSNorm) → FFN/MoE      → 残差连接
```

**源码：**

```python
class MiniMindBlock(nn.Module):
    def forward(self, hidden_states, position_embeddings, ...):
        residual = hidden_states
        # Pre-Norm + Attention + 残差
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, ...
        )
        hidden_states += residual
        # Pre-Norm + FFN + 残差
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
```

### 常考问题

**Q: Pre-Norm vs Post-Norm 的区别？**

Pre-Norm（先归一化再计算）：训练更稳定，梯度流动更好，大模型普遍使用。
Post-Norm（原版 Transformer）：理论上表达能力更强，但深层网络训练困难。

**Q: 为什么 Decoder-only 模型主流？**

- 训练目标统一（next token prediction），易于扩展
- 生成能力强，且可通过 prompt 完成理解任务
- KV Cache 机制使推理高效

**Q: 权重绑定是什么？有什么好处？**

```python
# MiniMind 中 embedding 和 lm_head 共享权重
self.model.embed_tokens.weight = self.lm_head.weight
```

好处：减少参数量，并且 embedding 和输出投影语义一致，有利于训练。

---

## 2. RMSNorm

### 原理

RMSNorm 相比 LayerNorm 去掉了均值中心化，只做缩放：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

**源码：**

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

### 常考问题

**Q: RMSNorm 对比 LayerNorm 的优势？**

1. 计算更快（省去均值计算）
2. 效果相近，甚至更好
3. 无 bias 参数，更简洁

**Q: 为什么要 `.float()` 再 `.type_as(x)`？**

防止 bfloat16/float16 精度不足导致数值不稳定，norm 计算在 float32 精度下进行后再转回原类型。

---

## 3. RoPE 旋转位置编码

### 原理

RoPE 将位置信息编码到 Q、K 的旋转变换中，使注意力分数自然包含相对位置信息。

对于位置 $m$，频率 $\theta_i$：

$$q_m' = q_m e^{im\theta}, \quad k_n' = k_n e^{in\theta}$$

则 $q_m' \cdot k_n' = q_m \cdot k_n \cdot e^{i(m-n)\theta}$，即注意力分数只依赖相对位置 $m-n$。

**源码：预计算 cos/sin 表：**

```python
def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**旋转应用：**

```python
def apply_rotary_pos_emb(q, k, cos, sin, ...):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### YaRN 长度外推

MiniMind 支持 YaRN（Yet another RoPE extensioN）扩展上下文长度：

```python
# config 中配置
self.rope_scaling = {
    "type": "yarn",
    "factor": 16,
    "original_max_position_embeddings": 2048,
    # 外推长度 = 16 * 2048 = 32768
}
```

YaRN 核心思想：对低频分量（慢变化）进行插值，对高频分量保持原样，用线性 ramp 函数 $\gamma$ 平滑过渡。

### 常考问题

**Q: RoPE 为什么比绝对位置编码好？**

1. 天然编码相对位置，外推能力更强
2. 无需在 Embedding 层加位置向量，计算上更灵活
3. 可以通过 YaRN、NTK 等方式扩展上下文长度

**Q: rope_theta（base）如何影响外推？**

base 越大，频率越低，低频分量的波长越长，模型能覆盖的相对位置范围越大。LLaMA3 将 base 从 10000 提升到 500000。MiniMind 默认 `rope_theta=1000000`。

---

## 4. GQA 分组查询注意力 & KV Cache

### GQA 原理

Multi-Head Attention (MHA)：Q、K、V 各有 `num_heads` 个头。
Grouped Query Attention (GQA)：K、V 只有 `num_kv_heads` 个头，Q 仍有 `num_heads` 个头，多个 Q 头共享同一组 K/V。

**源码：**

```python
class Attention(nn.Module):
    def __init__(self, args):
        self.n_local_heads = args.num_attention_heads      # 8
        self.n_local_kv_heads = args.num_key_value_heads   # 2（GQA）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 4

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
```

**repeat_kv 实现 KV 扩展：**

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
```

### KV Cache

推理时缓存历史 K、V，避免重复计算：

```python
# 推理时拼接历史 KV
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

KV Cache 定位当前位置：

```python
# 从 past_key_values 确定起始位置
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
position_embeddings = (
    self.freqs_cos[start_pos:start_pos + seq_length],
    self.freqs_sin[start_pos:start_pos + seq_length]
)
```

### 常考问题

**Q: GQA 的优势？**

减少 KV 头数量，显著降低 KV Cache 显存占用和推理时的 memory bandwidth 消耗，同时保持接近 MHA 的效果。

**Q: KV Cache 的显存计算？**

每层每个 token 的 KV Cache 大小：`2 × num_kv_heads × head_dim × bytes_per_param`。
例如 MiniMind（2 KV 头，64 head_dim，8 层，bfloat16）：每 token = `2 × 2 × 64 × 8 × 2 bytes = 4KB`。

**Q: MHA / GQA / MQA 的区别？**

| 类型 | KV 头数 | 内存 | 效果 |
|------|---------|------|------|
| MHA  | = Q 头数 | 最大 | 最好 |
| GQA  | 介于两者 | 中等 | 接近 MHA |
| MQA  | 1        | 最小 | 略下降 |

---

## 5. Flash Attention

### 原理

标准 Attention 需要存储 $O(N^2)$ 的注意力矩阵，Flash Attention 通过分块计算（tiling）避免完整矩阵的显存访问，时间复杂度不变但显存降至 $O(N)$。

**MiniMind 使用方式：**

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

# 推理/prefill 阶段且无特殊 mask 时启用
if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None ...):
    output = F.scaled_dot_product_attention(
        xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True   # 自动生成因果 mask
    )
else:
    # 手动实现：causal mask + softmax
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ xv
```

### 常考问题

**Q: Flash Attention 为什么快？**

关键在于减少 HBM（显存）读写次数。标准 Attention 需要将 $N\times N$ 矩阵写入 HBM，Flash Attention 在 SRAM 中完成分块计算，HBM 访问量从 $O(N^2)$ 降至 $O(N)$。

**Q: Flash Attention 2 / 3 的改进？**

- FA2：改善了并行策略，减少非矩阵乘法运算，序列维度并行
- FA3：针对 H100 Hopper 架构优化，异步流水线

---

## 6. SwiGLU 激活函数 & FFN

### 原理

标准 FFN：`FFN(x) = W2 * ReLU(W1 * x)`

SwiGLU（门控线性单元）：

$$\text{FFN}_{SwiGLU}(x) = W_{down} \cdot (\text{SiLU}(W_{gate} \cdot x) \odot W_{up} \cdot x)$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（Sigmoid Linear Unit）。

**源码：**

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        # intermediate_size = int(hidden_size * 8/3)，对齐到64的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN['silu']

    def forward(self, x):
        # gate * up，再 down
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )
```

### 常考问题

**Q: 为什么 intermediate_size 取 `8/3 * hidden_size`？**

SwiGLU 有 3 个矩阵（gate/up/down），参数量约为 `3 * hidden * inter`，要使总参数与标准 2 矩阵 FFN（`2 * hidden * 4*hidden`）相当，则 `inter ≈ 8/3 * hidden`。

**Q: SiLU 和 GELU 的区别？**

两者都是平滑的非线性激活，SiLU = $x \sigma(x)$，GELU ≈ $x\Phi(x)$（正态 CDF）。SiLU 计算更简单，效果相近，LLaMA/MiniMind 采用 SiLU。

---

## 7. MoE 混合专家模型

### 原理

每层 FFN 替换为多个专家网络，通过门控网络为每个 token 选择 Top-K 个专家，只激活部分参数。

**MiniMind MoE 架构（DeepSeek 风格）：**
- `n_routed_experts`：路由专家数（默认4）
- `num_experts_per_tok`：每个 token 激活专家数（默认2）
- `n_shared_experts`：共享专家数（默认1，所有 token 必过）

**门控网络源码：**

```python
class MoEGate(nn.Module):
    def forward(self, hidden_states):
        # 计算每个专家的得分
        logits = F.linear(hidden_states.view(-1, h), self.weight)
        scores = logits.softmax(dim=-1)
        
        # 选 Top-K 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        
        # 归一化 top-k 权重
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        
        # 计算辅助负载均衡损失（训练时）
        if self.training and self.alpha > 0.0:
            # seq_aux 方式：序列级别均衡
            ce = torch.zeros(bsz, self.n_routed_experts)
            ce.scatter_add_(1, topk_idx_for_aux_loss,
                            torch.ones(bsz, seq_len * aux_topk)).div_(seq_len * aux_topk / n_experts)
            aux_loss = (ce * scores.view(bsz, seq_len, -1).mean(dim=1)).sum(dim=1).mean() * self.alpha
        return topk_idx, topk_weight, aux_loss
```

**MoE 前向（训练 vs 推理）：**

```python
class MOEFeedForward(nn.Module):
    def forward(self, x):
        topk_idx, topk_weight, aux_loss = self.gate(x)
        if self.training:
            # 训练：repeat_interleave 并行计算所有专家
            x_rep = x.repeat_interleave(num_experts_per_tok, dim=0)
            y = torch.empty_like(x_rep)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x_rep[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        else:
            # 推理：按专家 batch，避免重复计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight)
        
        # 加共享专家输出
        for expert in self.shared_experts:
            y = y + expert(identity)
        return y
```

**推理优化（按专家批量处理）：**

```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()  # 将同一专家的 token 聚合
    tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i-1]
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_out = self.experts[i](x[exp_token_idx]) * flat_expert_weights[idxs[start_idx:end_idx]]
        expert_cache.scatter_add_(0, exp_token_idx.view(-1,1).repeat(1, x.shape[-1]), expert_out)
    return expert_cache
```

### 常考问题

**Q: MoE 负载均衡损失的作用？**

防止所有 token 都路由到同一专家（专家坍塌）。辅助损失 `aux_loss = alpha * sum(expert_fraction * expert_avg_score)` 鼓励各专家负载均匀。

**Q: 共享专家（Shared Expert）的作用？**

类似 DeepSeek-MoE 设计，共享专家捕获通用知识，路由专家捕获特定模式，分工更明确，效果更好。

**Q: MoE 推理的挑战？**

- 通信开销：多机 MoE 需要 All-to-All 通信
- 负载不均：不同专家接收 token 数差异大
- 显存：所有专家参数都要加载，但只激活部分

---

## 8. 预训练 Pretraining

### 训练目标

Causal Language Modeling（CLM）：给定前 n 个 token，预测第 n+1 个 token。

**损失计算（源码）：**

```python
# MiniMindForCausalLM.forward
if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个预测
    shift_labels = labels[..., 1:].contiguous()        # 去掉第一个 token
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100   # padding 位置不计算 loss
    )
```

**数据集构建：**

```python
class PretrainDataset(Dataset):
    def __getitem__(self, index):
        tokens = tokenizer(text, max_length=max_length-2, truncation=True).input_ids
        tokens = [bos_id] + tokens + [eos_id]   # 加首尾特殊 token
        input_ids = tokens + [pad_id] * (max_length - len(tokens))
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100  # padding 不计算 loss
        return input_ids, labels
```

### 训练关键超参

| 参数 | MiniMind 默认值 | 说明 |
|------|--------------|------|
| learning_rate | 5e-4 | 预训练初始 LR |
| batch_size | 32 | per-device |
| accumulation_steps | 8 | 等效 batch = 256 |
| grad_clip | 1.0 | 梯度裁剪 |
| dtype | bfloat16 | 混合精度 |
| optimizer | AdamW | 无 weight decay 设置 |

### 常考问题

**Q: 预训练和 SFT 的 label 有什么区别？**

预训练：所有 token 都参与 loss 计算（含 prompt）。
SFT：只有 assistant 回复部分计算 loss，prompt/system 部分 label=-100 不计算。

**Q: 为什么用 ignore_index=-100？**

CrossEntropy 的 `ignore_index` 参数会跳过这些位置的梯度计算，padding 和 prompt 部分不应影响模型参数更新。

---

## 9. SFT 监督微调

### 数据格式与 Label Masking

SFT 只对 assistant 回复计算 loss：

```python
class SFTDataset(Dataset):
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)  # 全部初始化为 -100（不计算 loss）
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                # 找到 <bos>assistant\n 的位置，之后到 <eos>\n 之间才计算 loss
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id)
            else:
                i += 1
        return labels
```

**Chat Template 示例：**

```
<|im_start|>system
你是 minimind<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant    ← bos_id 起始，开始计算 loss
你好！有什么可以帮助你的？<|im_end|>  ← eos_id 结束
```

### 常考问题

**Q: Full SFT 和 LoRA SFT 的区别？**

Full SFT 更新全部参数，效果更好但成本高；LoRA 只训练低秩矩阵，参数量少，适合资源受限场景。

**Q: 灾难性遗忘是什么？如何缓解？**

微调时模型过度拟合新任务，忘记预训练知识。缓解方法：小 LR（DPO 建议 ≤5e-8）、少训练轮次、EWC 正则、混合预训练数据。

---

## 10. LoRA 低秩适配

### 原理

对预训练权重 $W \in \mathbb{R}^{d\times k}$，增加低秩分解：

$$W' = W + \Delta W = W + B \cdot A$$

其中 $A \in \mathbb{R}^{r\times k}$，$B \in \mathbb{R}^{d\times r}$，$r \ll \min(d,k)$。

训练时冻结 $W$，只更新 $A$ 和 $B$。

**MiniMind 源码：**

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # A: 高斯初始化（破坏对称性）
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B: 全零初始化（确保训练开始时 ΔW=0）
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对方阵 Linear 层应用（q/k/v/o 投影等）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            setattr(module, "lora", lora)
            original_forward = module.forward
            # 猴子补丁：原始输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora
```

### 常考问题

**Q: B 为什么全零初始化，A 为什么高斯初始化？**

确保训练开始时 $\Delta W = B \cdot A = 0$，即 LoRA 初始等价于原模型，不引入初始偏差。A 高斯初始化是为了打破对称性，让梯度能正常流动。

**Q: LoRA rank 如何选择？**

rank 越大，表达能力越强，参数越多。通常 r=4~64，通用任务 r=8 足够，复杂任务可以更大。

**Q: QLoRA 是什么？**

LoRA + 量化（4-bit NF4）：将基础模型量化以节省显存，LoRA 权重保持 bf16，反向传播时反量化计算梯度。

---

## 11. DPO 直接偏好优化

### 原理

DPO 将 RLHF 中的 RM 训练 + PPO 优化合并为一步，直接从偏好数据优化模型。

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 $y_w$ 为 chosen（偏好回复），$y_l$ 为 rejected（拒绝回复），$\beta$ 控制偏离参考模型的程度。

**源码：**

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # 对 sequence 取平均 log prob
    ref_log_probs   = (ref_log_probs * mask).sum(1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(1) / seq_lengths.squeeze()

    batch_size = ref_log_probs.shape[0]
    chosen_ref   = ref_log_probs[:batch_size//2]
    reject_ref   = ref_log_probs[batch_size//2:]
    chosen_policy = policy_log_probs[:batch_size//2]
    reject_policy = policy_log_probs[batch_size//2:]

    # log ratio 差
    pi_logratios  = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**训练时冻结参考模型：**

```python
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)  # 参考模型不更新
```

### 常考问题

**Q: DPO 和 RLHF 的区别？**

RLHF 需要训练独立奖励模型，再用 PPO 优化，流程复杂。DPO 直接从偏好对数据中推导出等价的监督信号，无需奖励模型，训练更稳定简单。

**Q: beta 参数的作用？**

beta 控制 KL 惩罚强度。beta 越大，模型越不能偏离参考模型；beta→0，退化为纯偏好优化，可能遗忘。MiniMind 默认 beta=0.1，LR 建议 ≤5e-8。

**Q: DPO 的局限性？**

1. 需要 chosen/rejected 对，数据构建成本高
2. 可能出现 chosen 的概率反而下降的问题
3. 对参考模型质量敏感

---

## 12. GRPO / RLHF

### GRPO 原理（Group Relative Policy Optimization）

GRPO 是 DeepSeek-R1 使用的算法，核心思想：对同一 prompt 采样多个回复，用组内相对奖励计算优势值，无需 Critic 网络。

$$\mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)} \hat{A}_i - \beta D_{KL}(\pi_\theta || \pi_{ref})$$

**优势值计算（源码）：**

```python
# 对同一 prompt 的 G 个回复，计算组内相对优势
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**KL 散度惩罚（token 级别）：**

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1  # 近似 KL

per_token_loss = -(
    torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    - args.beta * per_token_kl
)
policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
```

**奖励函数（推理任务）：**

```python
def reasoning_model_reward(rewards):
    # 格式奖励：检查 <think>...</think><answer>...</answer> 格式
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    format_rewards = [0.5 if re.match(pattern, r, re.S) else 0.0 for r in responses]
    rewards += torch.tensor(format_rewards)

    # 标记计数奖励：各标记出现恰好1次
    def mark_num(text):
        reward = 0
        for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
            if text.count(tag) == 1: reward += 0.25
        return reward
    rewards += torch.tensor([mark_num(r) for r in responses])
    return rewards
```

### 常考问题

**Q: GRPO 对比 PPO 的优势？**

PPO 需要训练 Critic（价值网络），显存和计算量是 Actor 的 2 倍。GRPO 用同 prompt 多采样的组内相对奖励代替 Critic，无需额外模型，更简洁高效。

**Q: DeepSeek-R1 的 "Aha Moment" 是什么？**

通过纯 GRPO 训练（只用格式奖励和正确性奖励），模型自发出现了思维链推理行为，无需人工标注 CoT 数据。这说明 RL 可以涌现出复杂推理能力。

---

## 13. 知识蒸馏

### 原理

用大模型（Teacher）的软标签（Soft Target）指导小模型（Student）训练，比直接用硬标签效果更好。

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{KL}$$

**源码：**

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return (temperature ** 2) * kl  # T² 缩放保持梯度量级

# 总损失
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

### 常考问题

**Q: 温度 T 的作用？**

T 越大，Teacher 的概率分布越软（各类别概率差距缩小），包含更多"暗知识"（dark knowledge）。T=1 退化为标准交叉熵，T>1 让 Student 学习 Teacher 对相似 token 的相对概率。

**Q: 为什么蒸馏损失要乘以 T²？**

对 logits/T 求导时链式法则会产生 1/T 因子，乘以 T² 可以补偿，保持梯度量级与 T 无关。

---

## 14. 混合精度训练 & 梯度相关技巧

### 混合精度（AMP）

```python
dtype = torch.bfloat16  # 或 float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss / args.accumulation_steps

scaler.scale(loss).backward()

if (step + 1) % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

### 梯度累积

等效扩大 batch size，解决显存不足问题：

```
实际 batch_size = per_device_batch_size × accumulation_steps × num_gpus
MiniMind: 32 × 8 × 1 = 256（单卡等效 batch）
```

### 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

将所有参数梯度的全局 L2 范数裁剪到 1.0，防止梯度爆炸。

### 常考问题

**Q: bfloat16 vs float16 的区别？**

| | float16 | bfloat16 |
|--|---------|---------|
| 指数位 | 5 | 8（同 float32）|
| 尾数位 | 10 | 7 |
| 动态范围 | 小，易溢出 | 大，与 float32 相同 |
| 精度 | 高 | 低 |

bfloat16 不易溢出，无需 GradScaler，现代 GPU（A100/H100）优先推荐。float16 需要 GradScaler。

**Q: `zero_grad(set_to_none=True)` 的作用？**

`set_to_none=True` 直接释放梯度张量内存，比 `zero_()` 填零更省显存，且下次 backward 前 PyTorch 会重新分配。

---

## 15. 分布式训练 DDP

### MiniMind DDP 实现

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 单卡模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 模型包装
if dist.is_initialized():
    # 不同步 RoPE 频率表（它们是 buffer 而非参数）
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

**数据分片：**

```python
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
# 每个 epoch 打乱顺序
train_sampler.set_epoch(epoch)
```

### 常考问题

**Q: DDP vs DP 的区别？**

DP（DataParallel）：单进程多线程，有 GIL 限制，梯度在主卡汇总，存在负载不均。
DDP（DistributedDataParallel）：多进程，每卡独立前向+反向，用 All-Reduce 同步梯度，效率更高。

**Q: 为什么要忽略 freqs_cos/freqs_sin？**

这两个是 register_buffer（非可训练参数），DDP 默认也会同步 buffer，但它们在各卡上完全相同（由配置决定），同步是冗余的，排除可节省通信。

**Q: DDP 梯度同步时机？**

DDP 采用 bucket 机制，将参数分组，在反向传播时异步执行 All-Reduce，当一个 bucket 的梯度都计算完毕后立即同步，与后续反向传播并行，减少等待时间。

---

## 16. 学习率调度

### Cosine Decay with Warmup

MiniMind 使用余弦衰减：

```python
def get_lr(current_step, total_steps, lr):
    # 最小 LR = 0.1 * lr，峰值 LR = lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

| current_step | LR 值 |
|---|---|
| 0 | `lr * (0.1 + 0.45*2) = lr * 1.0`（峰值）|
| total_steps/2 | `lr * (0.1 + 0.45*1) = lr * 0.55` |
| total_steps | `lr * (0.1 + 0) = lr * 0.1`（最小值）|

### 常考问题

**Q: 为什么要 Warmup？**

训练初期参数随机，梯度波动大，大 LR 会破坏已有预训练权重（SFT 场景）。Warmup 从小 LR 逐渐增大，让模型平稳进入训练状态。

**Q: 余弦调度 vs 线性调度？**

余弦调度：后期 LR 降低更平滑，尾部大量时间在低 LR 精细优化，通常效果更好。线性调度：LR 均匀衰减，实现简单。

---

## 17. Tokenizer

### MiniMind Tokenizer

```
词表大小：6400（极小，专为中文设计）
BOS token id：1
EOS token id：2
中文 1 token ≈ 1.5~1.7 字符
```

### 常考问题

**Q: BPE vs WordPiece vs SentencePiece？**

- BPE：从字符开始，按频率合并最常见对，GPT 系列使用
- WordPiece：类似 BPE，但按最大化训练数据似然合并，BERT 使用
- SentencePiece：语言无关，直接处理原始 Unicode，不依赖分词，LLaMA/Gemma 使用

**Q: 为什么 tokenizer 词表大小很重要？**

词表太小：OOV 多，同等信息需要更多 token，序列更长，计算更慢。
词表太大：Embedding 和 lm_head 参数量激增，稀有 token 训练不足。
一般大模型词表 32K~128K 之间。

---

## 18. Agent 相关

### Function Calling / Tool Use

MiniMind 支持 Function Calling，在 SFT 数据中包含 tools 信息：

```python
def create_chat_prompt(self, conversations):
    # 检测是否有 functions 定义
    tools = conversations[0]["functions"] if (
        conversations and conversations[0]["role"] == "system"
        and conversations[0].get("functions")
    ) else None
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, tools=tools
    )
```

### 常考问题

**Q: ReAct 框架是什么？**

Reasoning + Acting：模型交替进行推理（Thought）和行动（Action），通过工具调用获取观察（Observation），循环直到得出最终答案。

**Q: RAG 检索增强生成的流程？**

1. 离线：文档分块 → Embedding → 存入向量数据库
2. 在线：用户问题 → Embedding → 相似度检索 → 召回 Top-K 文档 → 拼入 Prompt → LLM 生成

**Q: Agent 的核心组件有哪些？**

- **规划（Planning）**：任务分解，子目标设定，思维链推理
- **记忆（Memory）**：短期（上下文），长期（外部存储/RAG）
- **工具（Tools）**：API 调用、代码执行、搜索
- **行动（Action）**：与环境交互，执行决策

**Q: 如何解决 LLM 上下文长度限制？**

1. 滑动窗口：只保留最近 N 个 token
2. 摘要压缩：定期总结历史对话
3. RAG：将长文档转为检索，用时再取
4. 长上下文模型：YaRN / RoPE 扩展（MiniMind 支持外推到 32768）

**Q: CoT（思维链）为什么有效？**

将复杂问题分解为中间步骤，每步占用更少的 token 空间，类似于人类的草稿纸；同时中间步骤的注意力机制可以访问前面的推理结果，降低每一步的难度。

---

## 总结速查表

| 技术点 | 关键参数/公式 | MiniMind 实现 |
|--------|-------------|--------------|
| RMSNorm | $x / \sqrt{mean(x^2)+\epsilon} \cdot \gamma$ | float32 内计算 |
| RoPE | rotate_half + cos/sin | YaRN 支持 32K |
| GQA | n_heads=8, n_kv_heads=2, n_rep=4 | repeat_kv |
| Flash Attention | scaled_dot_product_attention | seq_len>1 时启用 |
| SwiGLU | silu(gate) * up, inter=8/3*hidden | bias=False |
| MoE | top_k=2, n_experts=4, shared=1 | aux_loss 均衡 |
| LoRA | rank=8, A~N(0,0.02), B=0 | 方阵 Linear |
| DPO | beta=0.1, LR≤5e-8 | ref_model 冻结 |
| GRPO | G组采样, 组内归一化优势 | token级KL惩罚 |
| 蒸馏 | loss=α*CE + (1-α)*T²*KL | T=温度 |
| AMP | bfloat16 推荐 | autocast+scaler |
| 梯度累积 | accum_steps=8 | 等效batch=256 |
| DDP | NCCL backend | DistributedSampler |
| LR调度 | Cosine, min=0.1*lr | get_lr函数 |
