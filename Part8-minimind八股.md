这是一个非常硬核且极其关键的复习阶段！在面试大模型底层算法时，面试官往往会采用**“自顶向下提问，自底向上手撕”**的策略。

为了让你能在白板面试中展现出降维打击的实力，我将结合大模型（如 MiniMind/Llama 架构）的源码实现，为你逐一拆解这些核心技术的**数学原理、八股话术以及关键代码实现**。

---

### 第一部分：架构篇（白板推导与手撕主战场）

#### 1. RoPE (旋转位置编码)

* **面试官问**：RoPE 的本质是什么？为什么它比绝对位置编码好？
* **满分话术**：
* “RoPE 的核心思想是**通过绝对位置的旋转操作，隐式地表达相对位置信息**。”
* “在数学上，它通过欧拉公式将 Token 的位置 $m$ 转化为复平面上的旋转角度 $m\theta$。当计算 Query 和 Key 的内积时，我们会发现 $\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m-n)$，即**内积的结果只与相对距离 $(m-n)$ 有关**。”
* “优势在于：1. 具备绝对位置编码的实现简单性；2. 具备相对位置编码的平移不变性；3. 通过调整旋转频率的底数（Base），天然具备优秀的**长文本外推能力（Extrapolation）**。”


* **源码精要 (刻在脑子里)**：
面试官常问：旋转矩阵是怎么作用到特征上的？（注意奇偶维度的交叉）
```python
def apply_rotary_emb(xq, xk, freqs_cis):
    # freqs_cis 形状: (seq_len, head_dim/2) 复数形式
    # 将 xq 和 xk 转换为复数表示
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 复数乘法完成旋转，然后转回实数
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

```



#### 2. RMSNorm vs LayerNorm

* **面试官问**：RMSNorm 相比 LayerNorm 去掉了什么？为什么能这么做？
* **满分话术**：
* “RMSNorm（均方根归一化）**去掉了 LayerNorm 中的均值平移（中心化）操作**，只保留了方差缩放。”
* “这背后的假设是：在深度神经网络中，特征的均值偏移对模型的非线性表达影响不大，真正起稳定梯度作用的是**方差的缩放**。”
* “这样做的好处是省去了计算均值的开销，使得前向和反向传播的速度提升了约 10%-20%，同时在准确率上与 LayerNorm 表现相当。”


* **源码精要**：
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # 1. 计算均方根 (Root Mean Square)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        # 2. 乘以可学习的缩放参数 weight (注意没有 bias)
        return self.weight * x_norm

```



#### 3. SwiGLU 激活函数

* **面试官问**：SwiGLU 公式是什么？为什么现在不用 ReLU 甚至 GELU 了？
* **满分话术**：
* “公式：$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$，其中 $\text{Swish}(z) = z \cdot \sigma(\beta z)$。”
* “它的核心优势在于引入了 **门控机制 (Gating Mechanism)**。它用一条线性变换的路径去控制另一条经过非线性激活的路径，这使得信息的流动更加平滑。”
* “相比 ReLU 的硬截断，SwiGLU 是平滑且处处可导的，这在训练深层 Transformer 时能提供更好的梯度流动，经验上收敛速度更快，困惑度（Perplexity）更低。”



#### 4. 高频手撕代码：Causal Multi-Head Attention (必写对 Mask)

* **避坑指南**：面试官主要看两点：1. `Q @ K.T` 后的除以 $\sqrt{d_k}$ 有没有忘；2. **下三角 Mask** 是不是加在了 Softmax 之前。
* **白板标准答案**：
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.size()

        # 1. 线性映射并切分多头 (Batch, seq_len, n_heads, d_k) -> 调换维度
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算 Attention Scores (除以 sqrt(d_k) 防梯度消失)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 生成下三角因果 Mask (Causal Mask)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
        # 在 Softmax 前，将未来信息替换为负无穷
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax 与 V 聚合
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # 5. 拼接多头并输出投影
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(context)

```



---

### 第二部分：训练篇（工程与炼丹的细节）

#### 1. 混合精度训练 (AMP) 与 Bfloat16

* **面试官问**：大家都用 AMP，为什么在预训练大模型时，业界普遍摒弃了 FP16 而转向 BF16？
* **满分话术**：
* “这取决于浮点数的比特位分配。FP16 只有 **5 位指数位**，表达的数值范围非常窄（最大约 65504），在大模型动辄几十亿参数的训练中，极易发生**梯度溢出（Overflow/Underflow）**，导致 Loss 变成 NaN。”
* “而 **Bfloat16 保留了和 FP32 一模一样的 8 位指数位**，牺牲的是尾数位（精度）。虽然计算结果不那么精确，但在深度学习中，梯度的一点点噪音相当于天然的正则化；但只要范围够大，就不会溢出，训练极其稳定。”



#### 2. SFT 阶段的 Loss Masking

* **面试官问**：SFT 训练时，数据是怎么构造的？为什么要对 Prompt 算 Loss 呢？如果不 Mask 会怎样？
* **满分话术**：
* “在代码中，SFT 数据的标签（Labels）在 Prompt 部分会被替换为 `-100`（PyTorch 的 CrossEntropyLoss 默认的 `ignore_index=-100`）。”
* “这是因为我们希望模型学习的是**‘如何回答问题’（拟合目标分布），而不是‘如何提出问题’（拟合用户分布）**。如果不对 Prompt 进行 Mask，模型在训练初期会浪费大量的模型容量（Capacity）去尝试预测用户会问什么话，这毫无意义，且会导致模型对指令的遵循能力大幅下降。”


* **源码精要 (数据构造逻辑)**：
```python
# 假设 token IDs 如下：[System] [User Query] [Model Answer]
# input_ids:  [101, 102, 23, 45, 67, 103, 88, 99, 100, 104]
# labels:     [-100,-100,-100,-100,-100, 103, 88, 99, 100, 104]

```



---

### 第三部分：对齐篇 - DPO 推导 (超高区分度大招)

由于你不使用 PPO（RLHF），面试官会死抠 **DPO (Direct Preference Optimization)**。如果能在白板上推出 DPO 的核心逻辑，你的技术面基本就稳了。

* **面试官问**：DPO 为什么叫“直接”偏好优化？它是怎么绕过奖励模型（Reward Model）的？能不能写一下它的 Loss？
* **白板推导步骤与话术**：
1. **引出传统 RLHF 的目标**：在 RLHF 中，我们要最大化奖励 $r(x, y)$，同时通过 KL 散度约束当前模型 $\pi_\theta$ 不要偏离参考模型 $\pi_{ref}$ 太远。
2. **写出最优策略闭式解**：根据数学推导（变分推断），上述目标的最优策略存在闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$


3. **核心魔法（变量代换）**：既然最优策略和 Reward 存在上述关系，我们对公式取对数，反向**把 Reward $r(x,y)$ 用语言模型的概率 $\pi(y|x)$ 表达出来**：

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$


4. **代入 Bradley-Terry (BT) 模型**：偏好数据中，人类认为 $y_{chosen}$ 比 $y_{rejected}$ 好的概率是 $P = \sigma(r(chosen) - r(rejected))$。
5. **推导最终 Loss**：把第 3 步的 $r(x,y)$ 代入第 4 步，神奇的事情发生了，配分函数 $Z(x)$ 直接被减掉了！我们得到了完全不需要显式 Reward Model 的交叉熵损失函数：

$$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)} \right)$$




* **一句话总结**：“DPO 的本质就是**用两个模型（当前模型和参考模型）输出概率的比值对数差，去等价替换原本的奖励差值**，从而把复杂的强化学习过程，降维成了一个简单的二元交叉熵分类任务。”



### 导师行动建议

将上述内容作为你的**“面试前 10 分钟必看清单”**。
对于手撕代码（特别是 MHA），建议你在一张白纸上默写 3 遍，确保你在没有自动补全的情况下，依然能写出完美处理各种维度的张量变换操作。

如果你觉得自己掌握了，可以尝试回答这个追加问题，测试一下肌肉记忆：**在手写 Self-Attention 时，最后一步 `context.contiguous().view(...)`，这里的 `contiguous()` 为什么是必须的？如果不写会报什么错？**