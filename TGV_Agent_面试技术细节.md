# TGV 缺陷检测工艺评估 Agent — 面试技术细节手册

> 覆盖：GraphRAG 原理 · vLLM 优化 · BGE-M3 检索 · 实体抽取 · 因果图谱建模

---

## 目录

1. [项目整体架构概述](#1-项目整体架构概述)
2. [GraphRAG 原理与二次开发](#2-graphrag-原理与二次开发)
3. [模型选型：Qwen3-14B + BGE-M3](#3-模型选型qwen3-14b--bge-m3)
4. [知识图谱构建：实体抽取与三元组建模](#4-知识图谱构建实体抽取与三元组建模)
5. [高并发推理引擎：从 Ollama 到 vLLM](#5-高并发推理引擎从-ollama-到-vllm)
6. [检索增强推理链路](#6-检索增强推理链路)
7. [Local Search vs Global Search 混合策略](#7-local-search-vs-global-search-混合策略)
8. [可解释性设计：推理结果溯源](#8-可解释性设计推理结果溯源)
9. [常见追问 Q&A](#9-常见追问-qa)

---

## 1. 项目整体架构概述

```
原始工艺文档 / 缺陷报告
        │
        ▼
  [Indexing Pipeline]
  TextUnit 切分 → 实体/关系抽取（Qwen3-14B）
        │
        ▼
  知识图谱存储（三元组/四元组）
  + Leiden 社区划分 + 社区摘要
        │
  ┌─────┴──────┐
  Local Search  Global Search
        │
        ▼
  [Query Pipeline]
  用户工艺评估查询 → 实体识别 → 子图检索
        │
        ▼
  结构化图数据 → 自然语言模板 → LLM 多轮推理
        │
        ▼
  可溯源推理结果（缺陷诊断 + 工艺优化建议）
```

**一句话总结给面试官：** 项目本质是把非结构化工艺文档转化为结构化因果知识图谱，再通过图检索 + LLM 推理，实现"问缺陷→定原因→给调参建议"的全链路 Agent。

---

## 2. GraphRAG 原理与二次开发

### 2.1 为什么选 GraphRAG 而不是普通 RAG？

| 对比维度 | Baseline RAG（向量检索） | GraphRAG |
|---|---|---|
| 知识表示 | 离散文本块，无关系 | 实体-关系图，有结构 |
| 多跳推理 | 弱，跨段落信息难关联 | 强，可沿图边遍历 |
| 全局理解 | 依赖 top-k 命中，容易遗漏 | 社区摘要覆盖全局 |
| 工业因果链 | 无法显式建模 A→B→C | 三元组天然表达因果 |

**TGV 场景的核心诉求**是"激光功率密度过高 → 热影响区扩大 → 孔壁粗糙度增加 → 开路缺陷"这类多跳因果链，Baseline RAG 中各段文本是孤立的，无法自动串联；GraphRAG 的图结构天然适配。

### 2.2 GraphRAG 索引流程（Index Pipeline）原理

GraphRAG 官方 Index Pipeline 分为以下核心步骤：

```
原始文档
  ↓ chunk_text
TextUnit（可配置 chunk_size / overlap）
  ↓ extract_graph（LLM抽取）
实体 Entity + 关系 Relationship + Claim（可选）
  ↓ build_communities（Leiden算法）
分层社区结构（Level 0/1/2...）
  ↓ summarize_communities（LLM摘要）
社区摘要 Community Report
  ↓ generate_embeddings（BGE-M3）
向量索引（实体 + 关系 + 社区摘要）
```

**Leiden 算法原理：**
- Leiden 是 Louvain 算法的改进版，解决了 Louvain 中社区内部连通性不保证的问题
- 目标函数是模块度（Modularity）：$Q = \frac{1}{2m}\sum_{ij}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)$
- 分三阶段迭代：局部节点移动 → 精化社区（子分区）→ 聚合图
- 图谱构建后按层级生成社区摘要，供 Global Search 使用

### 2.3 二次开发的核心改动点

**① 自定义实体类型（entity_types）**

原生 GraphRAG 默认抽取通用类型（PERSON, ORG, GEO 等），在工业场景中完全不适用。我们在 `prompts/entity_extraction.txt` 中自定义了：

```python
ENTITY_TYPES = [
    "EQUIPMENT",       # 设备：激光器型号、刻蚀机台
    "PROCESS_EVENT",   # 工艺事件：激光钻孔、湿法刻蚀、电镀
    "PROCESS_PARAM",   # 工艺参数：激光功率密度、刻蚀时长、温度
    "DEFECT_NAME",     # 缺陷名称：开路、孔锥度异常、孔壁粗糙
    "DEFECT_CAUSE",    # 缺陷原因：热影响区过大、刻蚀不均匀
    "PRODUCT",         # 产品：TGV基板、通孔
    "INSPECTION_METHOD"# 检测方法：SEM、X-ray、AOI
]
```

**② 四元组建模**

标准三元组 `(头实体, 关系, 尾实体)` 无法携带置信度或条件信息。我们扩展为四元组：

```
(PROCESS_PARAM:激光功率密度>5W/cm², CAUSES, DEFECT_NAME:孔壁粗糙, condition:材料=硼硅玻璃)
```

在图谱中以关系属性字段存储 `condition` 和 `confidence`，供后续推理过滤。

**③ 自定义抽取 Prompt**

针对工业文本特点（大量数字、单位、工艺术语），在 Prompt 中加入：
- Few-shot 示例（工艺参数边界触发缺陷的典型案例）
- 输出格式约束（JSON Schema 强制对齐实体类型）
- 否定关系处理（"功率低于3W时不会产生孔壁损伤"→ 负向关系边）

---

## 3. 模型选型：Qwen3-14B + BGE-M3

### 3.1 Qwen3-14B 选型原因

| 考量维度 | 说明 |
|---|---|
| **中文工业文本理解** | Qwen 系列在中文语料上预训练充分，对工艺术语、单位、数字理解优于同参数量西方模型 |
| **指令遵循能力** | Qwen3 支持 thinking mode，结构化输出（JSON）遵循率高，适合图谱抽取 |
| **本地部署可行性** | 14B 在单张 A100 80G 可 int4 量化部署，满足企业数据不出域要求 |
| **上下文窗口** | 支持 32K context，可容纳多个 TextUnit 同时抽取，减少 API 调用次数 |

### 3.2 BGE-M3 选型原因与原理

**BGE-M3（BAAI General Embedding - Multi-Functionality, Multi-Linguality, Multi-Granularity）**

三大核心能力：

```
Multi-Functionality：
  ├── Dense Retrieval（稠密向量，余弦相似度）
  ├── Sparse Retrieval（BM25-like 词频权重）
  └── Multi-Vector Retrieval（ColBERT-style，token级交互）

Multi-Linguality：100+ 语言，中英混排工业文本无需分语言处理

Multi-Granularity：支持 8192 token 长文本，适合工艺报告段落级检索
```

**为什么不用 text-embedding-ada-002 或 OpenAI 系列？**
- 工业数据涉密，不能调用外部 API
- BGE-M3 支持本地部署，且在中文检索基准（CMTEB）上显著优于 ada-002

**Dense + Sparse 混合检索的原理：**
- Dense：BERT-style Encoder，[CLS] 向量表示语义，解决同义词问题（"孔壁粗糙" ≈ "孔壁质量差"）
- Sparse：对 token 做激活权重预测，保留精确关键词匹配能力（"BGE-M3" 等专有名词不会被语义泛化）
- 混合分数：$score = \alpha \cdot score_{dense} + (1-\alpha) \cdot score_{sparse}$

---

## 4. 知识图谱构建：实体抽取与三元组建模

### 4.1 GraphRAG 的 Map-Reduce 抽取机制

GraphRAG 在抽取阶段使用 **Map-Reduce** 策略处理大文档：

```
大文档
  │
  ▼ (Map阶段，并行)
[TextUnit_1] → LLM抽取 → 局部三元组_1
[TextUnit_2] → LLM抽取 → 局部三元组_2
[TextUnit_n] → LLM抽取 → 局部三元组_n
  │
  ▼ (Reduce阶段)
实体去重合并（同一实体在不同文档中的多次提及→同一节点）
关系聚合（权重=共现次数 / 置信度平均）
  │
  ▼
最终知识图谱
```

**实体去重的关键**：采用 LLM 生成标准化实体名称 + 向量相似度双重校验，避免"激光功率密度"和"激光能量密度"被当作两个不同节点。

### 4.2 工艺-缺陷因果链建模示例

```
节点：
  - [PROCESS_PARAM] 激光功率密度 (id: pp_001)
  - [DEFECT_CAUSE]  热影响区扩大 (id: dc_001)
  - [DEFECT_NAME]   孔壁粗糙度异常 (id: dn_001)
  - [DEFECT_NAME]   开路缺陷 (id: dn_002)

边：
  pp_001 --[TRIGGERS {threshold: ">5W/cm²", confidence: 0.92}]--> dc_001
  dc_001 --[LEADS_TO  {confidence: 0.88}]--> dn_001
  dn_001 --[RESULTS_IN {confidence: 0.85}]--> dn_002

四元组表示：
  (激光功率密度, TRIGGERS, 热影响区扩大, ">5W/cm²")
```

### 4.3 为什么覆盖 5 万条三元组？

TGV 通孔制造涉及激光钻孔、湿法刻蚀、电镀、退火等多个工序，每道工序有 10~20 个关键参数，参数间存在交叉影响（如温度和刻蚀时长对同一缺陷的协同作用），文档来源包括工艺 SOP、缺陷分析报告、设备手册等异构文本，5 万条是覆盖主要因果路径所需的量级。

---

## 5. 高并发推理引擎：从 Ollama 到 vLLM

### 5.1 Ollama 的问题根因分析

在 Map-Reduce 抽取阶段，数百个 TextUnit 并发请求打满 Ollama 后端，暴露了三类问题：

**① KV Cache 显存碎片化**

Ollama 使用静态显存分配：为每个请求预分配一块连续显存存储 KV Cache（Key-Value Cache，即 Transformer 每层注意力的中间结果缓存）。

- 不同请求序列长度差异大（短的 512 token，长的 4096 token）
- 静态分配导致短请求浪费大块显存，长请求又可能 OOM
- 类比：像用固定大小的停车位停不同长度的车，总有浪费

**② 延迟持续劣化**

Ollama 的调度是 FCFS（先来先服务），请求串行或简单并行处理。随着并发增加，后续请求等待队列变长，P99 延迟线性增长，无法做请求间的动态填充。

**③ HTTP 超时中断**

Map 阶段单个 TextUnit 抽取耗时可能达到 30~60s，Ollama 的 HTTP keep-alive 超时设置与 LLM 推理时间不匹配，导致约 10 分钟后连接被服务端主动断开。

### 5.2 vLLM 的解决方案原理

#### PagedAttention（核心创新）

**原理**（类比操作系统虚拟内存分页）：

```
传统（Ollama/HuggingFace）:
  Request_A: [KV_block____连续____4096tokens________]  ← 预分配，浪费
  Request_B: [KV_block____连续____4096tokens________]  ← 实际只用512

vLLM PagedAttention:
  物理显存分成固定大小的 Block（如 16 tokens/block）
  Request_A: [Block_3][Block_7][Block_12]...  ← 按需分配，不连续也行
  Request_B: [Block_1][Block_9]...            ← 逻辑连续，物理分散

  Block Table（逻辑→物理地址映射）：
  Request_A: 逻辑0→物理Block_3, 逻辑1→物理Block_7...
```

**效果**：显存利用率从 ~60% 提升到 ~90%+，OOM 问题消除。

**Prefix Caching（KV Cache 复用）**：
- 多个请求共享相同的系统 Prompt（如我们的实体抽取 Prompt 模板），这部分 KV Cache 可以跨请求共享
- 显著减少重复计算，在 Map-Reduce 场景中大量 TextUnit 共享同一 System Prompt，收益明显

#### Continuous Batching（持续批处理）

```
传统 Static Batching：
  Batch = [Req_A(完成), Req_B(完成), Req_C(还差20token), 空槽]
  → 必须等 Req_C 完成才能处理新请求，GPU 空转

Continuous Batching（Iteration-level scheduling）：
  每个 decode step 后检查：
    - 已完成的请求 → 立即从 batch 中移除
    - 队列中等待的请求 → 立即填入空槽
  → GPU 利用率持续维持高位，吞吐量提升 2~4x
```

#### 指数退避重试策略

```python
import asyncio
import random

async def call_vllm_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await vllm_client.generate(prompt)
        except (RateLimitError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            # 指数退避 + 随机抖动，避免惊群效应
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

**为什么加随机抖动（Jitter）**：纯指数退避下，多个请求会在同一时刻重试（惊群），再次打满服务端。加 uniform(0,1) 随机抖动使重试时间错峰分布。

### 5.3 迁移效果对比

| 指标 | Ollama | vLLM |
|---|---|---|
| 显存利用率 | ~60% | ~90% |
| 并发吞吐量 | ~20 req/min | ~80+ req/min |
| OOM 频率 | 高（并发>8时） | 极低 |
| 超时中断 | 约10min后触发 | 无（连接管理优化） |
| 5万条知识抽取耗时 | >8h（含中断重启） | ~2h |

---

## 6. 检索增强推理链路

### 6.1 查询实体抽取：基于 BERT 联合解码

用户输入的工艺评估查询是自然语言，如：
> "激光功率密度调到 6W/cm² 后出现了孔壁粗糙的问题，可能是什么原因？"

需要从中识别出：
- 实体：`激光功率密度`（PROCESS_PARAM）、`孔壁粗糙`（DEFECT_NAME）
- 参数值：`6W/cm²`
- 查询意图：因果溯源

**BERT 联合解码模型架构：**

```
输入: "激光功率密度调到6W/cm²后出现了孔壁粗糙的问题"
  ↓
BERT Encoder → [CLS, T1, T2, ..., Tn]
  ↓                    ↓
意图分类头            NER 序列标注头
(因果溯源/参数查询     (B-PROCESS_PARAM / I-PROCESS_PARAM /
/优化建议)              B-DEFECT_NAME / O ...)

联合训练目标：
L = λ1 * L_intent + λ2 * L_ner
```

**为什么用联合解码而不是分两步做**：
- 实体类型识别依赖意图（同一个词"温度"，在"温度对缺陷的影响"vs"当前温度是多少"中，意图不同，检索策略不同）
- 联合训练让两个任务互相提供信号，参数共享 BERT Encoder，效率更高
- 单模型部署，延迟更低

### 6.2 子图挖掘

识别出目标实体后，在知识图谱中执行 K-hop 子图提取：

```python
def extract_subgraph(entity_ids, graph, k=2):
    """
    从目标实体出发，提取 k 跳范围内的子图
    用于因果链追溯（通常 k=2 覆盖"参数→直接原因→缺陷"三级链）
    """
    subgraph_nodes = set(entity_ids)
    subgraph_edges = []
    
    for hop in range(k):
        new_nodes = set()
        for node in subgraph_nodes:
            neighbors = graph.neighbors(node)
            # 过滤：只保留因果类关系边（TRIGGERS/LEADS_TO/CAUSES）
            causal_edges = [e for e in neighbors 
                          if e.relation_type in CAUSAL_RELATIONS]
            new_nodes.update([e.target for e in causal_edges])
            subgraph_edges.extend(causal_edges)
        subgraph_nodes.update(new_nodes)
    
    return subgraph_nodes, subgraph_edges
```

### 6.3 结构化图数据 → 自然语言模板

图数据是结构化的，LLM 理解结构化数据的效率远低于自然语言，需要做"图到文"的转换：

```python
# 三元组转自然语言模板
CAUSAL_TEMPLATE = (
    "根据工艺知识库，当{head_entity}达到{condition}时，"
    "会导致{relation_desc}，进而引发{tail_entity}。"
    "（数据置信度：{confidence:.0%}，来源文档：{source_doc}）"
)

# 示例输出：
# "根据工艺知识库，当激光功率密度达到>5W/cm²时，
#  会导致热影响区扩大，进而引发孔壁粗糙度异常。
#  （数据置信度：92%，来源文档：工艺SOP_v3.2_第14章）"
```

**为什么这一步很重要：**
- 直接把 JSON 格式的图数据塞给 LLM，模型需要先"解析"格式再理解内容，增加推理负担
- 自然语言模板消除了结构化数据与 LLM 自然语言理解之间的 **Representation Gap**
- 模板中嵌入置信度和来源，引导 LLM 在推理时考虑证据强度

---

## 7. Local Search vs Global Search 混合策略

### 7.1 Local Search（局部搜索）

**适用查询**：针对特定实体/参数的精确问题
> "孔壁粗糙的直接原因有哪些？"

**原理**：
1. 定位查询实体在图谱中的节点
2. Fan-out 到邻居节点（1~2跳）
3. 收集相关实体描述 + 关系 + 原始 TextUnit
4. 组装 Context 送入 LLM

**Context 构建优先级**（按 token budget 依次填充）：
```
1. 目标实体描述（最高优先级）
2. 直接相关关系
3. 关联实体描述
4. 相关社区摘要
5. 原始 TextUnit（最低优先级，兜底）
```

### 7.2 Global Search（全局搜索）

**适用查询**：需要跨多个工序、全局视角的问题
> "TGV 制造中最常见的缺陷类型及其共同根因是什么？"

**原理**：
1. 遍历所有社区摘要（由 Leiden 算法生成的层级摘要）
2. **Map 阶段**：LLM 对每个社区摘要独立生成局部答案 + 重要性打分
3. **Reduce 阶段**：按打分排序，选 top-k 社区摘要，LLM 综合生成最终答案

**为什么全局搜索要用 Map-Reduce**：
- 所有社区摘要的 token 总量远超 LLM context window
- Map-Reduce 把问题分而治之，每个 Map worker 只处理一个社区，天然并行
- Reduce 阶段只看高分摘要，避免低相关内容稀释答案质量

### 7.3 混合检索策略（项目实现）

```python
def hybrid_search(query, query_type):
    if query_type == "CAUSAL_ATTRIBUTION":
        # 因果溯源：优先 Local Search，获取精确的因果路径
        local_result = local_search(query, k_hop=2)
        if local_result.confidence < 0.7:
            # 本地图谱证据不足时，补充 Global Search
            global_result = global_search(query)
            return merge_results(local_result, global_result)
        return local_result
    
    elif query_type == "PROCESS_OPTIMIZATION":
        # 工艺优化：需要全局视角，优先 Global Search
        return global_search(query)
    
    elif query_type == "PARAMETER_QUERY":
        # 参数查询：精确匹配，Local Search
        return local_search(query, k_hop=1)
```

---

## 8. 可解释性设计：推理结果溯源

这是项目的核心亮点之一，也是工业端客户最关心的问题。

### 8.1 溯源链路设计

每条推理结果都附带完整的证据链：

```json
{
  "conclusion": "建议将激光功率密度控制在4.5W/cm²以下",
  "reasoning_chain": [
    {
      "step": 1,
      "claim": "当前功率密度6W/cm²超过阈值5W/cm²",
      "evidence_type": "graph_edge",
      "graph_path": "激光功率密度 --[TRIGGERS {threshold:>5W/cm²}]--> 热影响区扩大",
      "source_doc": "工艺SOP_v3.2.pdf, 第14章, 第3段",
      "confidence": 0.92
    },
    {
      "step": 2,
      "claim": "热影响区扩大导致孔壁粗糙度增加",
      "evidence_type": "graph_edge",
      "graph_path": "热影响区扩大 --[LEADS_TO]--> 孔壁粗糙度异常",
      "source_doc": "缺陷分析报告_2024Q3.pdf, 表3-2",
      "confidence": 0.88
    }
  ],
  "alternative_causes": ["刻蚀液浓度偏低", "基板预处理不足"],
  "recommendation_basis": "历史案例：调整功率至4.2W/cm²后，孔壁粗糙缺陷率从12%降至1.3%"
}
```

### 8.2 为什么可解释性对工业场景至关重要

- **责任归因**：工艺调参失误可能导致整批产品报废，工程师需要知道推荐依据，不能盲目信任黑箱
- **知识验证**：领域专家可以 review 每条图谱路径，发现图谱中的错误并反馈修正
- **合规要求**：半导体制造受 IATF 16949 等质量体系约束，决策记录需可审计

---

## 9. 常见追问 Q&A

**Q1：知识图谱的准确率怎么评估？**

A：采用三层评估：
1. **实体抽取 F1**：人工标注 500 条工艺文本，计算 Precision/Recall/F1（实体类型正确 + 边界正确）
2. **关系抽取准确率**：抽样 200 条三元组，领域专家判断因果关系是否成立
3. **端到端评估**：构造 50 个标准工艺查询（有 ground truth 答案），计算 Answer Correctness

**Q2：Qwen3-14B 抽取时有幻觉怎么处理？**

A：三个层面对抗幻觉：
1. **Prompt 层**：加入"如果文本中没有明确的因果关系，不要推断，输出空列表"的负向指令
2. **置信度过滤**：抽取时让 LLM 同时输出置信度分数，低于 0.6 的三元组不入库
3. **交叉验证**：同一关系在多个 TextUnit 中均被抽取到，才提升置信度；只在单处出现的关系标记为"待验证"

**Q3：5 万条三元组图谱的存储和查询怎么做的？**

A：GraphRAG 原生支持将图谱存为 Parquet 格式（边/节点表），向量索引存为 Lance 格式。查询时用 NetworkX 做子图遍历（适合中等规模），如果图谱规模继续扩大（>100万节点），可迁移到 Neo4j 或 Nebula Graph 做原生图查询（Cypher/nGQL），利用图数据库的索引加速多跳查询。

**Q4：为什么不直接用 Fine-tuning，而要做 GraphRAG？**

A：
1. **样本稀缺**：TGV 缺陷样本少，Fine-tuning 容易过拟合，泛化到新缺陷类型能力弱
2. **知识更新代价**：工艺参数会随设备维护、原材料批次调整，Fine-tuning 需要重新训练；GraphRAG 只需增量更新图谱节点/边，成本极低
3. **可解释性**：Fine-tuning 是参数化知识，无法溯源；GraphRAG 每个推理步骤都有明确的图谱路径支撑

**Q5：BGE-M3 的多向量检索（ColBERT-style）具体怎么工作？**

A：ColBERT-style 的多向量检索不把文档压缩成单个向量，而是保留每个 token 的向量表示：
- 查询端：`Q = [q1, q2, ..., qm]`，m 个 token 各一个向量
- 文档端：`D = [d1, d2, ..., dn]`，n 个 token 各一个向量
- 相似度：$score(Q,D) = \sum_{i=1}^{m} \max_{j=1}^{n} q_i \cdot d_j$（MaxSim 算子）
- 优点：能做 token 级的精细匹配，专有术语（如"孔壁粗糙度 Ra 值"）不会因为整体语义相似而被错误匹配到不相关文档

**Q6：项目最大的技术挑战是什么？**

A：最大的挑战是**工业因果关系的复杂性与 LLM 抽取能力之间的 Gap**。工艺文本中很多因果关系是隐式的（"调整功率后问题消失"→ LLM 需要推断因果，而非直接识别），或者是条件触发的（"在高温环境下，该参数才会引发缺陷"）。解决方案是设计精细的 Few-shot Prompt，配合四元组建模显式存储条件信息，同时在后处理阶段用领域专家校验高置信度三元组，形成人机协作的图谱构建流程。

---

*文档版本：2025.08 | 覆盖技术栈：GraphRAG · Qwen3-14B · BGE-M3 · vLLM · PagedAttention · Leiden · BERT NER*
