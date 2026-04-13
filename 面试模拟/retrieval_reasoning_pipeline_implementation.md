# 检索增强推理链路实现细节与演进记录

## 1. 文档目标

这份文档用于持续记录当前项目中“检索增强推理链路”的：

- 实现结构
- 关键代码位置
- 八股原理与设计动机
- 每一轮验证暴露的问题
- 对应修复与演进方向

后续如果继续补：

- BERT 监督训练版 Query NER
- type-aware linking
- intent-aware rerank
- 正式 HybridSearch 模块

都建议继续增量追加在这份文档里，而不是分散到临时笔记。

---

## 2. 目标链路

简历中的目标表述是：

> 针对工艺评估查询中长文本实体易遗漏的问题，设计 Query NER 联合解码机制，基于 BERT 抽取候选实体并经 Fuzzy Linking 对齐，与向量相似度检索结果按优先级融合；实现 2-3 hop 子图挖掘与 Top-K 路径排序；构建 HybridSearch 混合检索引擎，实现「Global 宏观主题定位 → 实体传递 → Local 精细子图检索 → 上下文融合截断」的多轮推理 Pipeline。

当前项目已经落到代码层的链路可以拆成两层：

### 2.1 当前已落地链路

```text
Query
-> Heuristic Query NER
-> Fuzzy Linking
-> 2-hop DFS Subgraph Mining
-> Graph Narrator
-> Local Retrieval Context
```

### 2.2 当前已补充的增强链路

```text
Query
-> BERT-assisted Query NER Prototype
-> Fuzzy Linking
-> 2-hop DFS Subgraph Mining
-> Graph Narrator
-> Retrieval Bridge Verification
```

### 2.3 目标工业版链路

```text
Query
-> Query Intent Router
-> BERT Query NER
-> Type-aware Fuzzy Linking
-> Global Search
-> Entity Transfer
-> Local Subgraph Retrieval
-> Context Fusion / Truncation
-> LLM Reasoning
```

### 2.4 面向关系问句的结构化检索计划

为了提升关系问句的准确性，当前链路已经开始从“相关上下文扩展”收敛到“结构化关系检索”。

当 query 属于关系问句时，内部不再只做普通全文检索，而是先语义化为：

```text
intent = relation_query
subject = A
object = B
target = direct_relation | shortest_relation_path
```

例如：

```text
What is the relationship between Scrooge and Marley?
```

内部结构化为：

```text
intent = relation_query
subject = SCROOGE
object = MARLEY
target = direct_relation | shortest_relation_path
```

这样后续检索的目标就从：

- 找“与 Scrooge / Marley 相关的主题背景”

变成：

- 先找 `SCROOGE <-> MARLEY` 的直接边
- 没有直接边时，再找最短桥接路径
- 最后才补背景实体

这一步是当前关系问句准确率提升的关键。

### 2.5 面向更强因果图的 schema / prompt 改造方向

为了让后续重建出来的 TGV 图谱更接近“可执行因果图”，当前已经开始同时改两层：

#### 抽取层

在 `prompts-tgv/extract_graph.txt` 中，关系类型不再只停留在弱关联层，而是优先明确为：

- `CAUSES`
- `LEADS_TO`
- `RESULTS_IN`
- `MITIGATES`
- `PRECEDES`

并且要求模型把 `relationship_type` 真正输出进关系结构。

#### 检索层

在 `subgraph_mining.py` 中，causal retrieval 已经开始从：

- 无向邻接扩图

往：

- 方向优先、逆向降权的 causal candidate generation

收敛。

一句话总结：

> 后续如果要重建“更强方向语义的图谱”，关键不是只改 prompt，也不是只改 retrieval，而是要让“定向关系 schema”和“方向优先检索”从一开始就对齐。

---

## 3. 当前关键代码位置

### 3.1 Query NER

- `scripts/custom_modules/query_entity_extractor.py`
  - 当前启发式 Query NER
  - 停用词过滤 + regex / title pattern + fuzzy linking

- `scripts/custom_modules/query_entity_extractor_bert.py`
  - 可跑的 BERT-assisted Query NER prototype
  - 结合：
    - heuristic candidates
    - regex IDs
    - terminology / aliases
    - BERT embedding scoring

### 3.2 Query Intent

- `scripts/custom_modules/query_intent_router.py`
  - 当前为接口骨架
  - 目标用于区分 relation / causal / compare / recommendation 等 query 类型

### 3.3 子图挖掘

- `scripts/custom_modules/subgraph_mining.py`
  - 2-hop DFS
  - Top-K 路径排序
  - 关系展平

### 3.4 叙事化

- `scripts/custom_modules/graph_narrator.py`
  - 将结构化图谱数据转成自然语言上下文

### 3.5 检索链路验证

- `scripts/test_retrieval_pipeline.py`
  - 当前完整检索测试套件
  - 已支持 `--ner-mode heuristic|bert`

- `scripts/verify_phase3.py`
  - Global + Local 的 Phase 3 验证脚本
  - 已接入 `--ner-mode heuristic|bert`
  - 仍存在与当前 graphrag 包版本的 API 漂移问题

- `scripts/verify_bert_query_ner.py`
  - BERT Query NER prototype 的独立验证脚本

- `scripts/verify_bert_retrieval_bridge.py`
  - 避开旧版 graphrag API 漂移，直接验证：
    - BERT NER
    - Linking
    - Subgraph
    - Narrator

---

## 4. 模块实现细节

## 4.1 Heuristic Query NER

### 做了什么

当前启发式版本的 Query NER 主要负责：

- 从 query 中抽显式实体
- 避免整句误抽
- 保留低时延

核心策略：

1. 引号文本优先
2. 大写单词匹配
3. 标题格式短语匹配
4. 停用词过滤
5. 长度限制
6. 结果去重

### 为什么这样设计

因为查询阶段是一个：

- 高频
- 低时延
- 高稳定性

的入口模块。

如果直接让 LLM 做 Query NER：

- 时延高
- 成本高
- 结果不稳定
- 容易把 query 本身压缩重写，造成信息损失

所以第一版先用轻量规则做“显式实体锚点抽取”是合理的。

### 当前短板

- 对长 query 的边界理解有限
- 对复杂工业术语变体的泛化能力不足
- 对中文 / 中英混合 query 的鲁棒性有限

---

## 4.2 BERT-assisted Query NER Prototype

### 当前定位

当前不是“完整监督训练的工业 BERT NER 模型”，而是：

> 一个可运行的、可接入现有检索链路的 BERT-assisted candidate extractor。

### 具体做法

在 `scripts/custom_modules/query_entity_extractor_bert.py` 中，当前实现分 4 层：

1. **候选生成**
   - heuristic extractor
   - regex ID pattern
   - terminology / alias scan

2. **候选打分**
   - 本地 `bert-base-chinese`
   - mean pooling
   - 与术语表 surface embedding 计算余弦相似度

3. **高置信规则优先**
   - `LOT-YYYYMMDD-X##` 直接识别为 `PRODUCTLOT`
   - `TOOL-##` 类 pattern 直接识别为 `TOOL`
   - exact / alias match 优先级高于 semantic match

4. **启发式兜底**
   - 当 query 不在当前工业词表域内时，保留 heuristic fallback
   - 避免对通用图谱 query 完全失效

### 为什么是这种实现

因为当前阶段最重要的是：

- 让 BERT 真正跑起来
- 真正接进现有链路
- 保持与已有 Heuristic / Fuzzy Linking 兼容

而不是一开始就做完整监督训练。

这是一种典型的工程演进路径：

```text
先做可跑原型
-> 再接入现有系统
-> 再通过评测发现噪声与边界问题
-> 最后再决定是否上监督训练
```

---

## 4.3 Fuzzy Linking

### 做了什么

当前 linking 逻辑在：

- `scripts/custom_modules/query_entity_extractor.py`

核心做法：

1. 先做包含关系匹配
2. 再做 `difflib.get_close_matches`
3. 返回知识库标准实体

### 为什么需要这一层

因为 Query NER 抽出来的是“query 中出现的名字”，而知识图谱里保存的是“标准实体名”。

工业场景里常见情况：

- `LASER01`
- `LASER-01`
- `laser tool 01`

本质上是同一个工具。

如果没有 Linking 层：

- NER 结果和图谱节点无法对齐
- 子图挖掘方向会偏
- 检索增强会失效

### 当前短板

- 还不是 type-aware
- 还没有 alias dictionary 的系统化重排逻辑
- 还没有 query intent aware 的候选排序

---

## 4.4 2-hop DFS 子图挖掘

### 当前实现

在 `scripts/custom_modules/subgraph_mining.py` 中：

- 先构建邻接表
- 从种子实体出发做 DFS
- 控制 `max_hops`
- 按路径总权重排序
- 取 Top-K

### 为什么用 DFS，不用 BFS

#### 工程解释

工艺分析、关系分析更关心“完整路径”，不是“把所有邻居扫出来”。

DFS 更适合：

- 顺着一条链往下看
- 找到较完整的 2-hop / 3-hop 解释路径

BFS 更适合：

- 全量邻域扩展
- 容易把大量弱相关节点一层层铺开

#### 八股原理

可以把它总结成一句：

> BFS 更像做覆盖，DFS 更像做追因。

### 为什么是 2-hop / 3-hop

因为：

- `1-hop` 太浅，很多 query 拿不到完整因果链
- `>3-hop` 噪声迅速增加

所以 2-hop / 3-hop 是一个经验上较合理的平衡点：

- 足够覆盖“缺陷 -> 原因 -> 参数 / 工序 / 设备”
- 又不会把图谱扩成不可控的巨大邻域

### 当前短板

- 当前排序还偏“路径权重总和”
- 对 relation query 缺少双实体约束
- 容易被高 degree hub 节点带偏

---

## 4.5 Graph Narrator

### 当前实现

在 `scripts/custom_modules/graph_narrator.py` 中：

- 将实体列表写成 `Key Entities`
- 将关系列表写成 `Relationships and Connections`

### 为什么需要 Narrator

图结构天然适合：

- 结构化检索
- 路径扩展
- 因果追踪

但大模型更擅长消费：

- 自然语言上下文
- 有线性叙事结构的证据

所以 Narrator 的意义不是“美化输出”，而是：

> 把结构化图谱转换成 LLM 更稳定可读的推理上下文。

### 当前短板

- 还没有 query-type-aware 模板
- relationship query 还会被叙事成“主题背景”
- 缺少针对 relation / causal / compare 的差异化模板

---

## 4.6 HybridSearch 目标架构

### 当前状态

目前仓库里已经有：

- Global Search 思路
- Entity transfer 思路
- Local Search + Subgraph + Narrator 的原型组合

但还没有一个完全成型、生产化的 `hybrid_search.py` 正式模块。

### 目标结构

```text
Query
-> Query Intent Router
-> Query NER
-> Fuzzy Linking
-> Global Search
-> Entity Transfer
-> Local Subgraph Retrieval
-> Context Fusion / Truncation
-> LLM Reasoning
```

### 为什么要这样分层

#### Global

负责：

- 宏观主题定位
- 找大范围相关社区
- 给 Local 提供先验方向

#### Entity Transfer

负责：

- 从 Global 结果里抽实体
- 和 Query NER 结果做融合

#### Local

负责：

- 细粒度子图扩展
- 路径和关系检索

#### Fusion

负责：

- 合并 Global 和 Local 上下文
- 控制 token 长度
- 防止局部证据淹没全局主题，或全局摘要盖掉局部路径

---

## 5. 八股原理整理

## 5.1 为什么不用 LLM 做 Query NER

### 回答

查询阶段强调：

- 高频
- 低时延
- 稳定

LLM 做 NER 的主要问题：

1. 时延高
2. 成本高
3. 结果不稳定
4. 容易重写 query，造成显式搜索词信息丢失

### 八股原理

> 查询理解模块优先保证确定性和时延，生成模型适合回答，不适合做高频入口解析。

---

## 5.2 为什么 Query NER 之后还要做 Fuzzy Linking

### 回答

NER 只负责抽出 query 中的候选实体文本；
Linking 才负责把候选实体映射到知识图谱里的标准节点。

### 八股原理

> NER 解决“抽什么”，Linking 解决“它是谁”。

---

## 5.3 为什么需要联合解码

### 回答

联合解码不是指一个模型同时做多个头，而是指三路信号协同参与检索：

1. Query NER 抽出的显式实体
2. Linking 后的标准实体
3. 向量检索召回的语义相关信息

### 八股原理

> 精确匹配负责锚点，语义检索负责补召回，二者结合才能兼顾 precision 和 recall。

---

## 5.4 为什么不用纯向量检索

### 回答

纯向量检索擅长语义相关，但不擅长稳定表达：

- 谁和谁有关系
- 哪个缺陷由哪个原因导致
- 哪条路径是更强证据链

### 八股原理

> 向量检索能找“像什么”，图检索更适合找“怎么连起来”。

---

## 5.5 为什么要做 Graph Narrator

### 回答

因为结构化图谱适合检索，但不一定适合直接让 LLM 消费。

### 八股原理

> 图是系统内部的推理结构，自然语言是大模型最稳的消费接口。

---

## 5.6 为什么要做 Global + Local

### 回答

如果只有 Local：

- 容易只看到局部节点和边

如果只有 Global：

- 容易只剩主题总结，缺少局部证据链

### 八股原理

> Global 负责“站得高”，Local 负责“看得细”。

---

## 4.7 Structured Retrieval Plan（关系问句专用）

### 当前改造目标

当前已经开始为 relation query 单独增加“结构化检索计划”层，而不是复用普通 query 的扩展逻辑。

### 当前实现位置

- `scripts/custom_modules/query_intent_router.py`
- `scripts/custom_modules/subgraph_mining.py`
- `scripts/custom_modules/graph_narrator.py`
- `scripts/verify_bert_retrieval_bridge.py`

### 具体做法

#### Step 1：Query Intent Router

先识别 query 类型。

对：

- `What is the relationship between A and B?`
- `How are A and B connected?`

这类问题，统一路由成：

- `relation_query`

并尽量解析出：

- `subject`
- `object`
- `target`

#### Step 2：Structured Plan

生成内部计划：

```text
intent = relation_query
subject = SCROOGE
object = MARLEY
target = direct_relation | shortest_relation_path
```

#### Step 3：Relation-aware Retrieval

优先级改成：

1. 找直接边
2. 找最短桥接路径
3. 再做局部背景补充

#### Step 4：Relation-specific Narration

Narrator 不再先堆背景，而是改成：

1. Direct Relationship
2. Shortest Bridge Path
3. Supporting Entity Context

### 为什么这是必要的

因为原来的检索逻辑虽然能把：

- `SCROOGE`
- `MARLEY`

都召回出来，但最终给出的仍然是“两个实体周围的背景知识”，而不是“它们之间的关系答案”。

关系问句本质上不是：

- “请给我一些和 A、B 相关的上下文”

而是：

- “请告诉我 A 和 B 之间的连接是什么”

这要求检索目标函数发生变化。

### 八股原理

> 对关系问句，检索目标应该从“相关性最大”切换到“连接性最强”。

---

## 4.8 Structured Retrieval Plan（因果问句专用）

### 当前改造目标

对于因果问句，系统不应该只返回“相关背景”，而应该优先回答：

- 谁是更强的原因候选
- 证据链怎么传递
- 哪条因果路径最值得优先展示

### 典型 query

例如：

```text
Why did Marley influence Scrooge's attitude towards Christmas?
```

或工业场景中的：

```text
Why did pulse_energy_drift lead to hole_wall_roughness?
What caused copper_fill_void in batch A03?
Why did rinse_conductivity anomaly trigger carryover_contamination?
```

### 结构化后的检索计划

当前将这类 query 语义化成：

```text
intent = causal_query
subject = cause_candidate
object = effect / focus
target = root_cause | causal_chain | supporting_evidence
```

例如：

```text
Why did Marley influence Scrooge's attitude towards Christmas?
```

会被内部解析成：

```text
intent = causal_query
subject = MARLEY
object = SCROOGE'S ATTITUDE TOWARDS CHRISTMAS
target = root_cause | causal_chain | supporting_evidence
```

### 当前实现位置

- `scripts/custom_modules/query_intent_router.py`
- `scripts/custom_modules/subgraph_mining.py`
- `scripts/custom_modules/graph_narrator.py`
- `scripts/verify_bert_retrieval_bridge.py`

### 具体做法

#### Step 1：识别因果问句

通过 query intent router 匹配：

- `why`
- `cause`
- `caused`
- `lead to`
- `result in`
- `due to`
- `influence`
- `affect`

#### Step 2：解析 subject / object

尽量从 query 中抽：

- cause candidate
- effect / focus

如果 query 本身没有明确边界，就回退到：

- linked entities
- extracted names

#### Step 3：因果路径重排

在 `subgraph_mining.py` 中新增：

- `rank_causal_paths(...)`

当前优先级是：

1. 路径描述中包含因果词汇
2. 路径更贴近 focus entity
3. hop 更短
4. 总权重更高

#### Step 4：因果 Narrator

Narrator 不再只写普通背景，而是改成：

- Causal Focus
- Strongest Supported Causal Chain
- Supporting Entity Context

### 八股原理

> 对因果问句，系统要从“谁和谁相关”切换到“谁导致了谁、证据链是什么”。

---

## 6. 验证记录与问题复盘

## 6.1 通用文本检索结果分析：Scrooge / Marley

问题：

> `What is the relationship between Scrooge and Marley?`

当时输出表现：

- 链路能跑通
- 速度快
- 上下文扩充显著

但核心问题是：

- 召回结果更偏主题扩展
- 中心节点容易主导路径排序
- 没有优先给出 Scrooge 和 Marley 的直接关系路径

暴露的问题：

1. 路径排序更偏全局权重，不够 relation-aware
2. hub node 会稀释双实体关系问答
3. Narrator 更偏背景说明，而不是关系证据

对应改进方向：

- 引入 relation-aware path ranking
- 引入双实体约束
- 为不同 query type 设计不同 Narrator 模板

---

## 6.2 BERT Query NER 原型：第一次可跑验证

脚本：

- `scripts/verify_bert_query_ner.py`

验证结果：

- 4 条 `TGV-mini` query
- 目标实体命中 `13/13 = 100%`

说明：

- 工业 query 上已经能稳定抽到：
  - LOT ID
  - TOOL ID
  - 参数名
  - 缺陷名

第一次暴露的问题：

- 出现噪声候选：
  - `TEM`
  - `copper_void`
- 原因是 prototype 会保留一部分“补召回式”相关候选

对应改进：

- 保留 exact / alias 命中优先
- pattern ID 直接高置信度通过
- 短大写 token 过滤

---

## 6.3 BERT 接入现有检索脚本

已做改造：

- `scripts/test_retrieval_pipeline.py`
- `scripts/verify_phase3.py`

新增能力：

- `--ner-mode heuristic`
- `--ner-mode bert`

设计原则：

- 默认不改变原有行为
- 显式切换时才启用 BERT prototype

这样做的好处：

- 便于做 A/B 验证
- 不影响当前已有实验结果

---

## 6.4 老版 GraphRAG API 漂移问题

在把 BERT 接入现有 `verify_phase3.py` 的过程中，暴露出多处历史兼容问题：

### 问题 1：旧版模块路径失效

现象：

- `graphrag.query.context_builder.graph_narrator` 找不到

原因：

- 当前项目实际使用的是仓库内 `scripts/custom_modules/graph_narrator.py`
- 老脚本还引用包内旧路径

改进：

- 改为统一从 `scripts.custom_modules` 引入

### 问题 2：indexer adapter API 漂移

现象：

- `read_indexer_communities` 不存在
- `read_indexer_entities` 参数签名与旧脚本不一致

原因：

- 当前 conda 环境里的 `graphrag` 版本和历史验证脚本写法不一致

改进：

- 部分兼容修复
- 但最终证明：老脚本不适合作为当前 BERT 接入的第一验证入口

### 问题 3：config 加载方式变化

现象：

- `load_config("settings.yaml")` 在当前版本下报错

原因：

- 当前版本要求传 root dir，而不是 yaml 文件路径

改进：

- 调整为 `load_config(".")`

### 问题 4：parquet schema 与 adapter 预期不完全一致

现象：

- `entities.parquet` 中缺少部分旧版适配器预期字段

原因：

- 生成产物 schema 与历史脚本写法不完全对齐

改进：

- 不再强行把 BERT 首轮验证绑死在老脚本上
- 转而新增桥接验证脚本

---

## 6.5 BERT Retrieval Bridge：当前最可靠的接入验证

新增脚本：

- `scripts/verify_bert_retrieval_bridge.py`

目标：

不依赖旧版 GraphRAG 包内部 API，直接验证当前项目真正控制的主链路：

```text
Query
-> BERT Query NER
-> Fuzzy Linking
-> Subgraph Mining
-> Graph Narrator
```

### 验证结果

命令：

```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_retrieval_bridge.py --ner-mode bert
```

针对通用 query：

> `What is the relationship between Scrooge and Marley?`

得到：

- 抽取：`['AOI', 'SCROOGE', 'MARLEY']`
- Linking：`['EBENEZER SCROOGE', 'GHOST OF JACOB MARLEY']`
- 子图：成功发现 10 条路径
- Narrator：成功生成 2826 字符上下文

### 这说明了什么

- BERT 模式已经真正接进“查询增强主链路”
- 不再只是独立 NER demo
- 能继续往 Linking / Subgraph / Narrator 传递

### 也说明了什么问题

- 当前 BERT prototype 的词表和 alias 主要偏 `TGV-mini`
- 对通用英文 `book` 图谱会有域不匹配噪声，如：
  - `AOI`

### 对应改进

新增 heuristic fallback：

- 在工业 query 中保留 BERT + lexicon 优势
- 在通用 query 中不至于完全失去 `SCROOGE / MARLEY` 这类显式实体

---

## 6.6 关系问句结构化检索：问题修复与验证

### 原始问题

在通用文本验证中，对于：

```text
What is the relationship between Scrooge and Marley?
```

系统虽然能召回：

- `SCROOGE`
- `MARLEY`
- `JACOB MARLEY`

但最终输出主要是：

- 相关实体背景
- 主题性上下文

而不是直接回答：

- 他们曾是 business partners
- Marley 死后以 ghost 形式成为 Scrooge 转变的催化剂

### 根因分析

1. Query NER 只识别实体，没有识别“关系问句意图”
2. 子图挖掘默认做 seed expansion，不优先找双实体之间的直接边
3. 路径排序偏相关性，不偏关系性
4. Narrator 默认先展开背景，而不是先输出关系结论

### 本轮改造

#### 改造 1：Intent + Structured Plan

在 `query_intent_router.py` 中新增：

- `subject`
- `object`
- `target`

并增加：

- `build_plan(query, extracted_names, linked_titles)`

#### 改造 2：Relation-aware retrieval

在 `subgraph_mining.py` 中新增：

- `find_direct_relationships(...)`
- `find_shortest_bridge_paths(...)`

排序逻辑变为：

1. 直接边优先
2. hop 更短优先
3. 权重更高优先

#### 改造 3：Relation-specific narrator

在 `graph_narrator.py` 中新增：

- `narrate_relation_answer(...)`

输出结构：

- Direct Relationship
- Shortest Bridge Path
- Supporting Entity Context

### 验证脚本

```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_retrieval_bridge.py --ner-mode bert --query "What is the relationship between Scrooge and Marley?"
```

### 验证结果

当前输出已经变成：

```text
intent=relation_query
subject=SCROOGE
object=MARLEY
target=direct_relation|shortest_relation_path

direct_edges=2
Direct 1: SCROOGE -> MARLEY (w=181.00)
Direct 2: MARLEY -> SCROOGE (w=66.00)
Bridge 1: SCROOGE -> MARLEY (w=181.00)
```

Narrator 现在会优先输出：

- **SCROOGE** and **MARLEY** are directly connected in the graph.
- Strongest edge: `SCROOGE -> MARLEY`
- Evidence: 他们曾经是 business partners，后来 Marley's ghost 成为 Scrooge 转变的重要触发因素

### 当前阶段结论

这次改造说明：

- 关系问句已经从“背景扩展式检索”切到“结构化关系检索”
- 输出开始对准 query 本身，而不是只堆相关上下文
- 这条设计对未来工业场景也非常关键，例如：
  - 设备 A 和缺陷 B 什么关系
  - 参数 X 和异常 Y 什么关系
  - 工序 P 与根因 C 的关系是什么

### 当前仍存在的问题

- 当前 BERT prototype 仍然会带少量域外噪声候选，例如 `AOI`
- shortest bridge path 还没有做 type-aware rerank
- Narrator 还没有 relation / causal / compare 的完整模板体系

---

## 6.7 因果问句结构化检索：第一次落地与验证

### 原始目标

把因果问句从：

- 普通相关性扩展

改成：

- 因果计划驱动检索
- 因果链优先排序
- 因果问句专用 Narrator 输出

### 本轮改造

#### 改造 1：intent router 扩展 causal plan

在 `query_intent_router.py` 中加入：

- `CAUSAL_PATTERNS`
- `subject / object / target`

并把因果问句统一收敛到：

```text
target = root_cause | causal_chain | supporting_evidence
```

#### 改造 2：causal path rerank

在 `subgraph_mining.py` 中加入：

- `rank_causal_paths(...)`

重排逻辑优先：

1. 因果词汇
2. focus entity 命中
3. hop 更短
4. 总权重更高

#### 改造 3：causal narrator

在 `graph_narrator.py` 中加入：

- `narrate_causal_answer(...)`

输出结构改成：

- Causal Focus
- Strongest Supported Causal Chain
- Supporting Entity Context

### 验证脚本

```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_retrieval_bridge.py --ner-mode bert --query "Why did Marley influence Scrooge's attitude towards Christmas?"
```

### 验证结果

当前输出已经能给出：

- `intent=causal_query`
- `subject=MARLEY`
- `object=SCROOGE'S ATTITUDE TOWARDS CHRISTMAS`
- 因果链优先输出

样例结果中，系统给出的 strongest supported chain 是：

```text
SPIRIT -> EBENEZER SCROOGE -> SPIRIT
```

并带有关系描述证据。

### 这次结果说明了什么

这说明：

- `causal_query` 结构化检索计划已经真正跑起来了
- 系统已经不再只是“把相关实体背景拼起来”
- 而是开始尝试输出“因果链 + 证据”

### 这次也暴露了什么问题

当前结果还不够理想，主要有三类原因：

1. **图谱域不匹配**
   - 当前验证图谱仍然是 `book` 通用图谱
   - BERT prototype 的词表和 alias 偏 `TGV-mini`
   - 所以会出现域外候选干扰，例如 `AOI`

2. **因果路径重排还不够强**
   - 当前只是基于因果词和 focus 命中做轻量重排
   - 还没有 type-aware / relation-type-aware 的强约束

3. **图谱本身存在关系噪声**
   - 通用文学图谱里的关系并不是严格工业因果图
   - 所以即使结构化检索方向对了，也可能选出“语义相关但并非最佳根因链”的路径

### 当前阶段结论

这轮因果改造的价值不在于“已经完美回答所有因果问句”，而在于：

- 检索目标函数已经从“相关扩展”切到“因果链解释”
- 系统已经具备了面向工业缺陷追因继续加强的正确架构

### 下一步应继续加强

1. type-aware causal path ranking
2. relation-type 优先级（CAUSES / RESULTS_IN / INFLUENCES）
3. causal query 的 focus slot 更准确抽取
4. 在 `TGV-mini` 图谱上做正式因果问句验证

---

## 7. 新一轮因果问句问题复盘：为什么 planning 对了，execution 还会偏

### 7.1 触发问题的样例

```text
Why did Marley influence Scrooge's attitude towards Christmas?
```

这条 query 的最新验证结果说明了一件很关键的事：

- `intent=causal_query` 识别是对的
- `subject=MARLEY`、`object=SCROOGE'S ATTITUDE TOWARDS CHRISTMAS` 也基本对
- 但后续 strongest causal chain 仍然容易偏成：

```text
SPIRIT -> EBENEZER SCROOGE -> SPIRIT
```

这说明系统已经能做 **query planning**，但 **retrieval execution** 还没有完全围绕 query 的因果焦点工作。

### 7.2 暴露出的真实问题

#### 问题 1：BERT Query NER 缺少 domain gating

在文学 query 中出现：

- `AOI`
- `via_offset`
- `seed_deposition`
- `deposition_uniformity`

这类工业术语，说明 BERT prototype 在 generic query 上仍会把候选错误投影到工业词表空间。

本质问题不是“BERT 完全不可用”，而是：

> 当前 prototype 同时挂着工业词表和 generic fallback，却没有在 query 进入时先判断领域。

#### 问题 2：causal retrieval 缺少 effect anchoring

虽然 structured plan 中已经有：

```text
object = SCROOGE'S ATTITUDE TOWARDS CHRISTMAS
```

但 retrieval ranking 之前只做了：

1. 因果词命中
2. focus entity 命中
3. hop 更短
4. 总权重更高

这还不够，因为它没有强约束：

- 路径是否覆盖 effect 相关锚点
- 路径是否同时贴合 `subject + effect`

于是 `SPIRIT` 这种高连接度节点仍然可能把主链抢走。

#### 问题 3：hub node 劫持

`SPIRIT` 在文学图谱中是高连接度节点。对 causal query 来说，如果只按“相关性 + 权重”排，系统就会自然偏向：

- 更容易连上的节点
- 不是更贴题的节点

这会导致结果从：

- “Marley 如何影响 Scrooge 对 Christmas 的态度”

退化成：

- “Scrooge 与 Spirit 之间有什么强相关路径”

### 7.3 本轮代码改造

#### 改造 1：BERT Query NER 增加 domain gating

在 `scripts/custom_modules/query_entity_extractor_bert.py` 中新增：

- query domain detection
- generic / industrial 两种抽取模式

当前规则是：

- 如果 query 中出现：
  - 工业 ID pattern
  - 工业术语 / alias 命中
  - 中文工业问句特征
  则进入 `industrial` 模式
- 否则进入 `generic` 模式

`generic` 模式下：

- 不再把普通候选强行投影到工业术语表做 semantic matching
- heuristic fallback 也会更保守，只保留 title-like 或显式 ID 候选

这一步的目标是：

> 先把跨领域污染压下去，避免工业词表把文学 query 带偏。

#### 改造 2：causal ranking 增加 effect anchoring

在 `scripts/custom_modules/subgraph_mining.py` 中升级了 `rank_causal_paths(...)`：

新增排序因素：

1. 是否覆盖 `subject`
2. 是否覆盖 `effect anchor`
3. `subject + effect` 是否同时出现
4. 因果词命中
5. 重复节点 / 近自环惩罚
6. hop 和总权重

也就是说，新的排序目标已经从：

- “路径强相关”

改成：

- “路径是否真正解释了 cause 如何影响 effect”

这一步是当前因果问句准确率最关键的改动。

#### 改造 3：桥接验证脚本接入新的因果重排参数

在 `scripts/verify_bert_retrieval_bridge.py` 中，`causal_query` 分支现在会把：

- `plan.subject`
- `plan.object`

同时传给 `rank_causal_paths(...)`，确保验证脚本跑到的是最新的 execution logic，而不是旧排序规则。

#### 改造 4：causal seed selection 补齐 effect 相关实体

本轮又定位到一个更底层的问题：

- `plan.object` 中其实已经出现了 `CHRISTMAS`
- linked entities 中也已经存在 `CHRISTMAS CAROL`
- 但旧版 causal retrieval 只取了 `linked_entities[:2]` 作为 seed

这会导致：

- effect 相关实体虽然被识别出来
- 却没有真正进入子图扩展
- 上层 rerank 只能在“不完整候选集”里做排序

所以这轮在 `verify_bert_retrieval_bridge.py` 中把 causal seed selection 改成：

1. 优先放入 `subject` 对应实体
2. 再放入 effect focus 相关实体
3. 最后补其余 linked entities

这一步的核心原理是：

> 如果 effect 侧节点没有进入 seed 集合，那么 effect anchoring 再强，也只能在错误的候选路径空间里做局部优化。

### 7.4 改造后的直接验证结果

#### 关系问句回归验证

```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_retrieval_bridge.py --ner-mode bert --query "What is the relationship between Scrooge and Marley?"
```

结果：

- Extracted names: `['SCROOGE', 'MARLEY']`
- Linked entities: `['EBENEZER SCROOGE', 'GHOST OF JACOB MARLEY']`
- direct edges: `2`
- strongest direct edge:

```text
SCROOGE -> MARLEY (w=181.00)
```

说明这轮 domain gating 没有破坏已经修好的 relation-query 路径。

#### 因果问句回归验证

```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_retrieval_bridge.py --ner-mode bert --query "Why did Marley influence Scrooge's attitude towards Christmas?"
```

改造前：

- 会出现 `AOI / via_offset / seed_deposition / deposition_uniformity`
- strongest path 容易被 `SPIRIT` 劫持

改造后：

- Extracted names: `['MARLEY', 'SCROOGE', 'CHRISTMAS']`
- Linked entities: `['GHOST OF JACOB MARLEY', 'EBENEZER SCROOGE', 'CHRISTMAS CAROL']`

这说明 domain gating 已经有效压掉跨领域词表串扰。

当前 strongest chain 仍不是最终理想答案，例如：

```text
EBENEZER SCROOGE -> JACOB MARLEY -> JACOB MARLEY
```

但相比改造前，路径已经明显开始围绕：

- `MARLEY`
- `SCROOGE`
- `CHRISTMAS`

而不是被 `SPIRIT` 这类 hub node 抢占主链。

继续补上 causal seed selection 之后，新的 strongest chain 已经进一步收敛为：

```text
SCROOGE -> GHOST OF JACOB MARLEY -> MARLEY
```

这说明当前系统已经至少完成了两步纠偏：

1. 从“工业词表污染”回到干净的文学 query 实体集合
2. 从 “SPIRIT 抢主链” 回到 `MARLEY / SCROOGE` 主轴附近

当前残留问题则变成更细的一层：

- `CHRISTMAS` 还没有真正进入 strongest chain
- Narrator 仍主要信任 top-1 path，而不是综合 top-k 支撑证据

继续往下做时，又补了两层关键改动：

#### 改造 5：relation-type-aware causal rerank（轻量版）

只靠“描述里有因果词”还不够稳，因为像 `SPIRIT` 这种节点的描述天然更像“因果解释”，会把 strongest path 再抢回去。

所以本轮在 `rank_causal_paths(...)` 中，把排序优先级继续收紧成：

1. 节点层 `subject` 命中
2. 节点层 `effect` 命中
3. 强因果词命中
4. 描述层的因果词命中
5. 重复节点惩罚
6. hop 和总权重

这一步的八股原理是：

> 对 causal query，节点层的 query 对齐要优先于描述层的“像因果”。

#### 改造 6：top-k consensus narration

在 `graph_narrator.py` 中，`narrate_causal_answer(...)` 不再只输出 top-1 path，而是：

- 先给 strongest supported causal chain
- 再补 `Additional Supporting Causal Paths`

这让最终回答从“单路径依赖”变成“主路径 + 支撑路径”的结构。

#### 改造 7：节点层包含式命中 + 文学因果语义加权

继续回归时又定位到一个细节问题：

- query 中的 effect 侧词是 `CHRISTMAS`
- 图里真实节点往往是 `CHRISTMAS CAROL`

如果节点层只做严格相等匹配，那么 effect 其实没有真正命中。  
所以本轮把节点层命中从：

- `term == node_title`

改成：

- `term in node_title`
- 或 `node_title in term`

同时，又为文学图谱补了一层更贴近“影响链”的叙事因果词加权，例如：

- `visit`
- `ghost`
- `warn`
- `christmas eve`
- `redemption`
- `transformation`

一句话八股原理：

> effect grounding 不能只看严格等值；在文学或工业图谱里，经常需要允许“概念词”命中“更完整的标准节点名”。

#### 再次回归验证结果

补上这两层后，同一条 query 的 top-3 进一步收敛为：

```text
SCROOGE -> GHOST OF JACOB MARLEY -> MARLEY
SCROOGE -> CHRISTMAS CAROL -> MARLEY
SCROOGE -> EBENEZER SCROOGE -> MARLEY
```

最关键的变化是：

1. `SPIRIT` 不再重新抢回 strongest chain
2. `CHRISTMAS CAROL` 已经进入 Additional Supporting Causal Paths，说明 effect 相关支撑路径开始进入最终叙事

继续补上“节点层包含式命中 + 文学因果语义加权”之后，结果保持稳定为：

```text
SCROOGE -> GHOST OF JACOB MARLEY -> MARLEY
EBENEZER SCROOGE -> MARLEY -> MARLEY
SCROOGE -> GHOST OF JACOB MARLEY -> CHRISTMAS
```

这说明：

1. `SPIRIT` 没有重新劫持主链
2. `CHRISTMAS` 稳定进入 Additional Supporting Causal Paths
3. strongest path 与 supporting paths 已经形成“主轴解释 + effect 补充”的结构

#### 改造 8：causal answer synthesis

到这一轮为止，Narrator 虽然已经有了：

- strongest path
- supporting paths

但最终输出依然更像“图路径展示”，还不像真正的 `why-answer`。  
所以本轮在 `graph_narrator.py` 中继续补了一层：

- 从 top-k causal paths 中抽 explanation fragments
- 先合成一条 `Why-Answer Synthesis`
- 再保留 strongest path 和 supporting paths 做证据支撑

当前 synthesis 的结构是：

1. 先说明这是多跳解释链，而不是单一直接边
2. 再指出 strongest currently supported path
3. 最后抽取最可复用的 explanation fragment

例如当前回归结果里，Narrator 顶部已经会先输出：

```text
Why-answer summary: The graph suggests that MARLEY influences SCROOGE'S ATTITUDE TOWARDS CHRISTMAS through a multi-step explanation chain rather than a single direct edge...
```

这一步的八股原理可以记成：

> Graph retrieval 负责找证据，answer synthesis 负责把多条局部证据压成一个更接近用户问句形式的解释。

这意味着当前 causal pipeline 已经从：

- `path retrieval`

往前推进成：

- `path retrieval + answer synthesis`

#### 改造 9：方向优先、逆向降权的 causal candidate builder

上面的所有改造，仍然默认路径候选来自“无向邻接近似图”。  
这会带来一个底层问题：

- top path 更像“相关路径”
- 不是严格的“因果方向路径”

所以本轮又在 `subgraph_mining.py` 中新增了：

- `build_directed_causal_candidates(...)`

它的行为是：

1. 优先沿图中已有的 `source -> target` 方向扩展
2. 仍允许逆向边进入候选集以保召回
3. 但逆向边会被显式标记，并在 rerank 中做降权

同时，在 `verify_bert_retrieval_bridge.py` 中，`causal_query` 已经改成优先走这条方向优先的 candidate builder。

这一步的八股原理可以记成：

> 如果想让 causal retrieval 更像因果图，而不是普通相关图，不能只在 rerank 上补权重，还必须在 candidate generation 阶段就体现方向偏好。

#### 抽取层同步收紧：关系类型改为更明确的定向因果类型

仅仅改检索层还不够，因为如果抽取层输出的边类型本身太弱，方向偏好也只能在“弱语义边”上做优化。  
所以本轮也同步收紧了 `prompts-tgv/extract_graph.txt`：

- 明确要求把 `source_entity -> target_entity` 当成定向边
- 对因果关系优先使用更明确的类型：
  - `CAUSES`
  - `LEADS_TO`
  - `RESULTS_IN`
  - `MITIGATES`
  - `PRECEDES`
- 只有在文本只支持弱关联时，才退回 `RELATED_TO`

这意味着项目后续如果重建工业图谱，边的语义会更接近“可执行因果图”，而不是“仅有 source/target 字段的关系表”。

#### 这轮回归结果说明了什么

在方向优先、逆向降权的 builder 接入后，同一条 query 的 top-1 变成了：

```text
EBENEZER SCROOGE -> MARLEY -> SCROOGE
```

并且 synthesis 中已经明确出现：

- `Marley's Ghost visits Scrooge to warn him ...`

这说明：

1. 检索链路已经开始显式利用方向性
2. `MARLEY -> SCROOGE` 这种更接近因果传播的边，开始在最终答案里占据更核心位置

但这轮也反过来证明了另一件事：

> 如果抽取层还没有把因果边类型建模得足够强，方向优先检索也可能沿着“方向存在但语义仍偏弱”的边走出绕回路径。

所以这轮结果很适合用来说明：

- **检索层的方向偏好已经开始起作用**
- **但最终要把 causal retrieval 做成真正的“因果图检索”，还必须继续收紧抽取层的关系 schema**

### 7.5 当前阶段结论

这轮改造的意义在于把系统从：

- **planning 正确，但 execution 漫游**

往前推成：

- **planning 正确，execution 开始围绕 cause + effect 收敛**

还没有完全解决的问题包括：

1. linking 层仍缺更强的低置信拒识机制
2. hub node 惩罚还可以继续加强
3. Narrator 目前仍然更依赖 top-1 path，后续可以升级为 top-k 共识式 summarization
4. effect 相关节点虽然已进入 seed 集合，但还需要 relation-type-aware rerank 才能进一步把 `CHRISTMAS` 拉进 strongest chain

但从架构上看，这一轮已经把 causal query 的主矛盾抓准了：

> 先做 domain gating，再做 effect anchoring，因果问答的主链才有机会回到真正的问题中心。

---

## 7. 当前阶段结论

当前这条检索增强推理链路已经可以准确拆成 3 层：

### 第 1 层：已稳定落地

- Heuristic Query NER
- Fuzzy Linking
- 2-hop DFS Subgraph Mining
- Graph Narrator

### 第 2 层：已补可跑原型

- BERT-assisted Query NER prototype
- 已接入现有检索脚本参数切换
- 已通过 `TGV-mini` NER 验证
- 已通过 retrieval bridge 验证

### 第 3 层：待继续工程化

- Query Intent Router 接入主链路
- type-aware linking
- relation / causal / compare 的路径排序
- Global + Local 的正式 HybridSearch 模块
- 面向工业 query 的专门评测

---

## 8. 后续扩展建议

### 8.1 Query NER

- 增加监督训练版 BERT NER
- 区分中文 / 英文 / 中英混合 query
- 引入 type-aware candidate rerank

### 8.2 Linking

- 引入 alias dictionary
- 引入 type-aware matching
- 引入 context-aware rerank

### 8.3 Subgraph

- 引入 relation-aware ranking
- 引入双实体关系 query 的 shortest-path / constrained-path 逻辑
- 对 hub node 加惩罚

### 8.4 Narrator

- 增加：
  - relation 模板
  - causal 模板
  - compare 模板
  - recommendation 模板

### 8.5 HybridSearch

- 正式实现 `scripts/custom_modules/hybrid_search.py`
- 将：
  - Global Search
  - Entity Transfer
  - Local Search
  - Context Fusion
  做成标准可调用模块

---

## 9. 建议的后续记录格式

后续每次验证都建议按下面模板追加：

### [日期 / 轮次]

**目标：**

**涉及模块：**

**验证脚本：**

**输入 query / 样例：**

**结果：**

**暴露问题：**

**根因分析：**

**修复内容：**

**是否已完成闭环：**

---

## 10. 一句话总结

这条链路当前已经从“只有文档设计”推进到：

> **启发式检索增强链路已稳定落地，BERT Query NER 原型已可运行并成功接入查询增强桥接链路；后续重点从可跑原型继续升级到工业级 HybridSearch 正式实现。**
