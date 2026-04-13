# TGV 缺陷检测工艺评估 Agent 面试问答版

## 1. 项目一句话介绍

### 问题
你先用一句话介绍一下这个项目。

### 回答

这个项目本质上是一个面向 TGV 基板通孔制造场景的因果知识增强工艺评估系统，通过 GraphRAG 把工艺文档、设备日志、缺陷报告和专家经验组织成可检索、可追溯、可解释的知识图谱，再结合大模型完成缺陷追因、工艺评估和优化建议生成。

### 追问

- 你为什么定义成 Agent，而不是问答系统
- 这个项目主要解决的是检测问题还是工艺问题
- 为什么强调"因果知识增强"

### 注意事项

- 不要上来就讲模型名字，先讲业务价值
- 强调"工艺评估、缺陷追因、优化建议"，不要只说知识问答
- "因果"二字要突出，这是区别于普通 RAG 的核心

---

## 2. 业务背景与项目价值

### 问题

这个项目的业务背景是什么，为什么有价值？

### 回答

TGV 场景里，工艺参数与缺陷类型之间存在复杂非线性因果关系。激光功率密度、刻蚀时长、设备状态等多因素耦合，导致缺陷形态多变。通用大模型缺乏领域专业知识，直接推理易产生误导性输出；同时缺陷样本稀缺，难以支撑有监督检测。

传统方式依赖专家经验或规则库，能解决一部分固定问题，但很难应对跨文档、多跳、条件复杂的追因场景。

所以这个项目的价值在于：
1. 把分散知识结构化，构建 5 万级工艺-缺陷知识库
2. 通过检索增强和图谱约束让模型输出更可解释、更可靠
3. 辅助工程师快速定位根因，缩短工艺调优周期

### 追问

- 为什么不用人工规则库继续做
- 这个场景的"复杂非线性"具体体现在哪
- 为什么普通 FAQ 检索不够
- 缺陷样本稀缺怎么解决

### 注意事项

- 价值描述要落到"帮助工程师追因和决策"
- 不要说成完全替代专家，更稳妥的表述是"专家辅助系统"
- 强调"因果"和"可解释"是核心竞争力

---

## 3. 为什么选择 GraphRAG

### 问题

为什么你们选择 GraphRAG，而不是普通 RAG？

### 回答

普通 RAG 更适合"某段文本里是否直接有答案"这类问题，但 TGV 场景的问题往往是跨文档、多跳、因果链式的。

例如"某批次孔壁粗糙上升，优先排查激光参数还是后段刻蚀"，答案并不在单一片段里，而需要关联：
- 工艺规范中的参数阈值
- 设备日志中的异常记录  
- 缺陷追溯报告中的归因分析
- 专家经验中的典型案例

GraphRAG 的优势是先把设备、工艺步骤、参数、缺陷、原因、措施组织成关系网络，再围绕问题做子图召回和证据整合，这样更适合工艺追因和评估。

**量化收益**：我们的实现中，子图挖掘平均带来 **+847% 上下文扩充**（最高 **+1705%**），显著提升了证据覆盖度。

### 追问

- 你说的多跳问题具体举个例子
- GraphRAG 相对普通 RAG 最大的收益是什么
- 为什么不直接做知识图谱查询而是还要加 LLM
- 上下文扩充率是怎么计算的

### 注意事项

- 回答重点放在"跨文档、多跳、全局理解"
- 不要空泛说"GraphRAG 更高级"，要结合场景解释
- 可以提具体数字增强说服力

---

## 4. 数据清洗与标准化

### 问题

你说构建了数据清洗流程，具体清洗了什么？

### 回答

在真实工业数据不足的阶段，我先用通用文本样例把清洗链路实现并验证，主要处理六类问题：

| 清洗步骤 | 处理内容 | 工业场景价值 |
|---------|---------|-------------|
| BOM 移除 | 去除 UTF-8 BOM | 避免首段文本污染 |
| 换行统一 | CRLF/CR → LF | 保证跨平台切块一致性 |
| 模板过滤 | 去除页眉页脚、版权说明 | 减少无效 chunk |
| 空白压缩 | 清理行尾空格、压缩空行 | 提高段落边界判断稳定性 |
| 段落去重 | 相邻重复段落规范化去重 | 减少重复实体和关系抽取 |
| 术语归一 | 设备/缺陷/参数名称标准化 | 降低实体碎片化 |

**验证结果**：以 `book.txt` 为例，原始 185,067 字符，清洗后 165,395 字符，清洗率 10.6%，主要是噪声内容。

**TGV 场景扩展**：针对工艺数据额外增加了参数单位归一（如 `W/cm²` 统一）、设备编号对齐、缺陷名称标准化、工序名称多语言映射等规则。

### 追问

- 清洗在 GraphRAG 工作流的哪个位置
- 段落去重怎么实现
- 术语归一的字典怎么构建
- 清洗后效果怎么量化评估

### 注意事项

- 强调"先用通用样例验证，再迁移到工业数据"的方法论
- 清洗是图谱质量的前置条件，不是附属工作
- 可以提具体数字（清洗率 10.6%）

---

## 5. vLLM 推理引擎优化

### 问题

你说把推理服务从 Ollama 迁移到 vLLM，具体解决了什么问题？

### 回答

**原生 Ollama 的问题**：在图谱 Map-Reduce 抽取阶段出现：
- KV Cache 显存碎片化（长时间运行后利用率下降）
- 延迟持续劣化（批处理场景尾延迟升高）
- 约 10 分钟后 HTTP 超时中断
- 平均吞吐约 300 tok/s

**vLLM 优化方案**：

1. **架构升级**：采用 PagedAttention + Continuous Batching，显存池化管理
2. **投机解码**：使用 Qwen2.5-Coder-7B 作为 draft model（词表 152064 与 14B 完全匹配），`num_speculative_tokens=5`
3. **稳定性策略**：指数退避重试、超时控制、并发限制、健康检查

**关键问题解决**：原始 0.5B draft 模型词表（151936）与 14B（152064）不匹配，改用 7B draft 后完全对齐。

**性能提升**：
```
优化前 (Ollama):    300.634 tok/s
优化后 (vLLM):      416.637 tok/s
提升幅度:           +38.6%
```

### 追问

- 你是怎么定位到 KV Cache 问题的
- 为什么抽取任务比聊天更容易触发这个问题
- 投机解码的原理是什么
- 7B 和 14B 词表匹配为什么重要
- 多卡隔离怎么做的

### 追问：多卡隔离怎么做的

**回答**

多卡隔离不是简单把模型“放到多张卡上”，而是把**生成服务**和 **embedding 服务**按职责拆成两个独立后端，再分别绑定不同 GPU。

**代码层面**主要有两步：

1. **vLLM 生成服务单独绑卡**
   - 在 `vllm.sh` 里通过 `CUDA_VISIBLE_DEVICES` 控制 vLLM 只能看到指定 GPU
   - 例如生成侧使用：
   ```bash
   CUDA_VISIBLE_DEVICES=1,2 \
   MODEL_PATH=/mnt/data2/ycl/graphrag/Qwen2.5-Coder-14B-Instruct \
   TENSOR_PARALLEL_SIZE=2 \
   bash vllm.sh
   ```
   - 这样 vLLM 只会使用 1、2 号卡提供 `http://127.0.0.1:8000/v1` 的 completion/query 服务

2. **GraphRAG 配置里把生成和 embedding 分开路由**
   - 在 `settings.vllm.yaml` 中：
     - `default_completion_model` 和 `query_completion_model` 指向 `http://127.0.0.1:8000/v1`
     - `default_embedding_model` 指向 `http://127.0.0.1:11434/v1`
   - 这意味着：
     - 生成请求走 vLLM
     - embedding 请求走 Ollama

**运行层面**，我最终采用的是：

- `vLLM`: 负责生成，固定到 `GPU 1,2`
- `Ollama`: 负责 embedding，固定到 `GPU 0`

这样做的原因是我在实验里确实遇到过资源争抢：

- 如果 vLLM 和 Ollama 共用同一张卡
- completion 看起来能正常返回
- 但 embedding 会在尾部阶段卡住，GraphRAG 停在 `generate_text_embeddings`

把两类服务拆到不同 GPU 后，问题就收敛了：

- vLLM 主生成链路稳定完成
- embedding 侧不再和生成侧抢显存
- 后续再把 embedding 模型从 `bge-m3` 换成 `mxbai-embed-large` 后，全流程才最终闭环

**一句话总结**：
多卡隔离的本质不是为了“看起来更高级”，而是为了让生成和 embedding 两条负载解耦，避免共享同一张 GPU 导致长时运行下的显存竞争和接口卡死。

### 注意事项

- 用现象倒推定位问题（延迟上升→超时→缓存管理）
- 不要说"我修改了 vLLM 源码"，除非真改过
- 可以说"利用其推理服务机制和调度能力提升吞吐"

---

## 6. Map-Reduce 机制详解

### 问题

GraphRAG 索引阶段的 Map-Reduce 是怎么工作的？你为什么需要优化它？

### 回答

**Map-Reduce 工作流程**：

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG 索引 Map-Reduce                  │
├─────────────────────────────────────────────────────────────┤
│  Map Phase (并行)                                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ Chunk 1 │  │ Chunk 2 │  │ Chunk N │  ...                 │
│  │ 抽取实体 │  │ 抽取实体 │  │ 抽取实体 │                     │
│  │ 抽取关系 │  │ 抽取关系 │  │ 抽取关系 │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       └─────────────┼─────────────┘                         │
│                     ↓                                        │
│            中间结果（实体/关系列表）                           │
│                     ↓                                        │
│  Reduce Phase (聚合)                                         │
│  ┌─────────────────────────────────────────┐                │
│  │ 实体去重与归一（相同实体合并描述）         │                │
│  │ 关系汇总与去重（相同关系合并证据）         │                │
│  │ 构建全局图谱（节点+边）                   │                │
│  │ 社区发现与摘要（Leiden算法）              │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

**为什么需要优化**：

1. **请求数量多**：一份文档切成几十到几百个 chunk，每个 chunk 都要调 LLM 抽取
2. **文本长度差异大**：有的 chunk 几百 token，有的几千 token，混合调度难度大
3. **运行时间长**：5 万条知识抽取可能需要持续运行数十分钟到数小时
4. **失败代价高**：某个 chunk 失败可能导致整个批次需要重跑

**我的优化策略**：

| 优化点 | 具体做法 | 效果 |
|-------|---------|------|
| 推理后端 | Ollama → vLLM | 吞吐 +38.6% |
| 并发控制 | 限制 `max_num_seqs=4` | 避免队列堆积 |
| 重试机制 | 指数退避 + 超时控制 | 短时抖动恢复 |
| 缓存复用 | 启用 GraphRAG cache | 断点续传 |
| 健康检查 | 启动前确认服务可用 | 减少无效请求 |

**监控指标**：
- Map 阶段：每秒处理 chunk 数、平均抽取延迟、失败率
- Reduce 阶段：实体去重率、关系合并率、社区划分质量

### 追问

- Map 阶段失败一个 chunk 怎么处理
- Reduce 阶段怎么判断两个实体是同一个
- 社区发现算法是什么，起什么作用
- 如果 Map 阶段并发太高会有什么问题

### 追问：Map 阶段失败一个 chunk 怎么处理

**回答**

这个问题要分成**单次请求层**和**整轮索引层**两层来看。

**第一层：单次请求层**

在当前配置里，GraphRAG 的 completion 和 embedding 都启用了重试：

- `settings.vllm.yaml` 里 completion/query 使用 `retry: exponential_backoff`
- embedding 模型也使用 `retry: exponential_backoff`

所以一个 chunk 对应的一次 LLM 请求如果只是短时抖动，例如：

- 瞬时超时
- 连接错误
- 后端短暂 500

会先走**指数退避重试**，而不是立刻判整个 workflow 失败。

**第二层：整轮索引层**

如果某个 chunk 在重试后还是失败，GraphRAG 的当前 workflow 一般不会“跳过这个 chunk 继续往后”，而是：

- 当前 workflow 记为失败
- 日志里能看到具体失败的阶段
- 但已经成功写入的 cache 和中间产物仍然保留

这也是为什么我前面一直强调：

- 不要轻易 `--clean`
- 优先保留 `cache/` 和已完成的 `output/`
- 修复问题后再续跑

比如我之前在 embedding 阶段遇到 `NaN/500` 时，并不是把前面 `extract_graph` 和 `community_reports` 全部重做，而是：

- 保留已有产物
- 只修 embedding 后端
- 再重新跑索引

这样已经成功的 chunk 结果可以复用，失败的阶段重新执行即可。

**工程上可以怎么讲**

> Map 阶段单个 chunk 失败时，先依赖请求级重试处理短时波动；如果重试后仍失败，当前 workflow 会停下来，但前面已成功的缓存和中间结果不会丢。我实际处理时会保留 cache 和 output，修复故障点后续跑，而不是每次全量重跑。

### 追问：Reduce 阶段怎么判断两个实体是同一个

**回答**

在原生 GraphRAG 里，这一步更接近**归一化聚合**，而不是完整意义上的工业级实体对齐系统。

它的核心思路是：

1. **Map 阶段先从不同 chunk 抽出局部实体**
   - 每个 chunk 都可能抽出自己的实体列表
   - 同一个实体会在多个 chunk 里重复出现

2. **Reduce 阶段按规范化后的实体名做聚合**
   - 例如同名或高度一致的 `title`
   - 再把不同 chunk 中关于它的描述汇总

3. **后续通过 `summarize_descriptions` 合并描述**
   - 也就是把多个局部描述整理成一个全局描述
   - 最终形成实体表里的：
     - `title`
     - `description`
     - `frequency`
     - `degree`

所以它本质上依赖的是：

- 名称一致性
- prompt 抽取风格一致性
- 前置清洗和术语归一

而不是一个特别复杂的 learned entity resolution 模型。

**为什么前面的清洗和术语标准化很重要**

因为如果同一个工业对象被写成不同形式，例如：

- `Laser-01`
- `LASER01`
- `laser tool 01`

那 Reduce 阶段就可能把它们当成不同实体，或者至少增加后续合并成本。

这也是为什么我在工业化改造里会特别强调：

- 术语归一
- 设备命名标准化
- 参数单位统一
- 别名字典维护

这些工作本质上是在帮助 Reduce 阶段更稳定地把“同一个东西”聚到一起。

**更准确的表述方式**

> 原生 GraphRAG 的 Reduce 阶段主要是基于实体名称和描述的一致性做聚合，再通过描述摘要形成全局实体表示。它不是一个完整的工业实体对齐系统，所以我在二开时会把术语归一、别名映射和 schema 约束前移，尽量在进入 Reduce 之前就减少同物异名。

### 注意事项

- Map-Reduce 是 GraphRAG 索引的核心机制，要讲清楚
- 强调"批量、长时间、高并发"场景的特殊性
- 可以画上面的流程图辅助说明

---

## 7. 项目解决的核心问题

### 问题

这个项目解决的是缺陷检测、工艺追因还是问答系统？

### 回答

表面上它是一个问答入口，但核心目标更偏**工艺追因**和**辅助决策**。

工程师通常不是不知道缺陷现象，而是不确定：

- 这个缺陷优先关联哪道工序
- 应优先排查哪些参数或设备
- 历史相似案例是怎么处理的
- 最终可以给出什么工艺调整建议

所以这个系统本质上是把缺陷问答、因果追因和工艺评估整合在一起，输出格式统一为：

```markdown
## 结论
高概率原因或评估结果

## 依据
- 证据链1：对应关系路径
- 证据链2：历史相似案例
- 证据链3：专家规则引用

## 建议
1. 优先排查项
2. 优先调整项
3. 风险提示
```

### 追问

- 为什么不把它定义成知识问答系统就好
- 它最终输出给工程师的是什么
- 工程师怎么验证建议的正确性

### 注意事项

- 最终输出要强调"结论 + 依据 + 建议"
- 不要只说"回答问题"，要说"辅助工艺决策"
- 三段式输出格式可以现场展示

---

## 8. GraphRAG 二次开发做了什么

### 问题

你说基于开源 GraphRAG 二次开发，具体改了什么？

### 回答

我主要做了四类改造。

**第一，schema 定制**。把通用实体改造成工业实体：
- 设备 (Tool)、工艺事件 (ProcessEvent)、工艺参数 (ProcessParameter)
- 缺陷名称 (DefectType)、缺陷原因 (DefectCause)、措施 (ActionMeasure)
- 检测结果 (DetectionEvent)、专家规则 (ExpertRule)

**第二，抽取 prompt 改造**。强调：
- 只抽原文明确支持的信息（保守抽取）
- 不确定就返回空，不要补常识
- 关系需要保留条件约束（四元组表示）
- 输出必须满足 schema 结构
- 尽可能绑定证据来源

**第三，检索增强链路改造**。设计了 Query NER 联合解码机制：
```python
# 启发式 NER + Fuzzy Linking
extracted = extractor.extract(query)  # < 1ms
linked = fuzzy_link(extracted, entities_dict)  # ~8ms
```

**第四，性能优化**。从 Ollama 迁移到 vLLM，吞吐提升 38.6%。

### 追问

- schema 是怎么设计出来的
- 你说的四元组多出来的一维是什么
- 你是不是改了 GraphRAG 源码，改在哪一层

### 注意事项

- 如果没有深改底层源码，不要讲成重写 GraphRAG
- 用"定制 prompt、定制 schema、定制链路"这种表述最稳

---

## 9. Query NER 联合解码机制

### 问题

你简历里提到设计 Query NER 联合解码机制，具体是什么？

### 回答

针对工艺评估查询中长文本实体易遗漏的问题，我把 Query NER 联合解码设计成了**双版本架构**：

1. **当前落地版**：轻量启发式 Query NER + Fuzzy Linking  
   - 目标是保证查询阶段高频、低时延、稳定
   - 这是当前仓库里已经落地并用于实验的版本

2. **演进版**：BERT 候选实体抽取 + Fuzzy Linking + 检索融合  
   - 目标是更好处理长 query、多实体、复杂边界和工业术语变体
   - 这是为了让整条链路更贴近“Query NER 联合解码”的正式工业方案

也就是说，当前代码里已经完整落地的是：

- 候选实体抽取
- Fuzzy Linking
- 与检索链路融合

而 BERT 版是我后续补齐到正式工业实现时的升级方向，接口和链路已经按这个方向收拢。

### 当前落地版：轻量 Query NER

当前用于实验验证的是轻量方案：

**阶段一：Query NER 实体抽取（改进版）**
```python
class QueryEntityExtractor:
    """基于启发式 + 停用词过滤 + 多策略匹配的实体抽取"""
    
    # 300+ 停用词（常见虚词 + 动词 + 噪声名词）
    STOP_WORDS = {
        'THE', 'A', 'AN', 'THIS', 'THAT', 'WHAT', 'WHICH', 'WHO',
        'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'HAVE', 'HAS', 'HAD',
        'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'SHOULD', 'COULD',
        'GET', 'MAKE', 'TAKE', 'COME', 'KNOW', 'THINK', 'LOOK',
        'TO', 'OF', 'IN', 'FOR', 'ON', 'WITH', 'AT', 'FROM', 'AS',
        'INTO', 'THROUGH', 'BEFORE', 'AFTER', 'BETWEEN', 'AND', 'OR', 'BUT',
        'WAY', 'THING', 'THINGS', 'PEOPLE', 'YEAR', 'YEARS', 'DAY', 'TIME',
        'RELATIONSHIP', 'RELATIONSHIPS', 'CONNECTION', 'CONNECTIONS',
    }
    
    def extract(self, query: str) -> list[str]:
        """三策略融合：引号 + 大写单词 + 标题格式"""
        query_clean = query.upper()
        
        # 策略1: 引号文本（最可靠）
        quoted = re.findall(r'"([^"]{2,30})"', query)
        
        # 策略2: 大写单词（限制长度 2-20 字符，避免整句匹配）
        upper_words = re.findall(r'\b[A-Z][A-Z]{1,20}\b', query_clean)
        
        # 策略3: 标题格式（首字母大写）
        titles = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        titles = [m.upper() for m in titles if len(m) > 2]
        
        # 合并 + 停用词过滤 + 长度过滤 + 去重
        candidates = quoted + upper_words + titles
        filtered = [c for c in candidates 
                   if c not in self.STOP_WORDS 
                   and 3 <= len(c) <= 30
                   and not c.isdigit()]
        
        return list(dict.fromkeys(filtered))
```

- 延迟：**~0.2ms**
- 覆盖：缺陷类型、工艺参数、设备名称、工序名称
- 关键改进：**停用词过滤**解决整句误抽问题

**阶段二：Fuzzy Linking 实体对齐**
```python
def get_entity_by_name_fuzzy(entities_dict, name, threshold=0.8):
    """精确匹配优先 + 模糊匹配兜底"""
    candidates = [(e.title, e) for e in entities_dict.values()]
    
    # 精确匹配（包含关系）
    for title, ent in candidates:
        if name in title.upper() or title.upper() in name:
            return [ent]
    
    # 模糊匹配
    titles = [t for t, _ in candidates]
    matches = difflib.get_close_matches(name, titles, n=3, cutoff=threshold)
    return [ent for title, ent in candidates if title in matches]
```

- 延迟：**~2ms**
- 作用：将抽取的别名/缩写对齐到知识库标准实体名

**阶段三：融合检索**
- NER 结果（精确匹配）优先级高于语义检索
- 构建查询锚点，指导子图召回方向

### 演进版：BERT Query NER + Fuzzy Linking

如果按简历里的正式表述继续往前收敛，推荐的升级方式不是推翻当前链路，而是在 Query NER 前端增加一个 BERT 候选抽取器。当前仓库里我已经补了一个**可跑的 BERT-assisted prototype**：

- 代码：`scripts/custom_modules/query_entity_extractor_bert.py`
- 验证脚本：`scripts/verify_bert_query_ner.py`
- 模型：本地 `bert-base-chinese`
- 当前定位：**BERT 辅助候选抽取 + 领域词表约束 + Fuzzy Linking**
- 口径说明：这是 runnable prototype，不是已经完成监督训练的工业终版 NER

```python
class BertQueryEntityExtractor:
    """
    目标角色：
    1. 从长 query 中抽取 span 级候选实体
    2. 输出候选文本 + span 边界 + 类型 + 置信度
    3. 将候选传给 Fuzzy Linking 做标准实体对齐
    """

    def extract_spans(self, query: str) -> list[EntitySpan]:
        ...
```

推荐链路会变成：

```text
原始 Query
→ BERT 抽候选实体 span
→ Fuzzy Linking 对齐标准实体
→ 与向量检索结果按优先级融合
→ 进入子图挖掘与 HybridSearch
```

这样当前代码和简历口径就能对齐成：

- 当前原型：启发式 Query NER 先验证链路可行性与时延
- 当前增强版：BERT-assisted prototype 负责更复杂 query 的候选抽取与候选打分
- 正式工业版：在 prototype 基础上补齐监督训练、type-aware linking 与系统评测

### 联合解码里的“联合”具体体现在哪里

“联合”不是指一个模型里同时解两个任务，而是指**三路信息联合参与后续检索**：

1. Query NER 抽出的候选实体
2. Fuzzy Linking 对齐后的知识库标准实体
3. 向量检索召回的语义相关实体 / 文本块

实际融合原则是：

- **NER 精确命中的实体优先级最高**
- **Fuzzy Linking 作为别名/缩写纠偏层**
- **向量检索负责补召回**

所以 Query NER 不会替代原始 query，而是给 HybridSearch 提供高置信度锚点。

**关键修复案例**：
```
查询: "What is the relationship between Scrooge and Marley?"
修复前（旧 regex）: ['WHAT IS THE RELATIONSHIP BETWEEN SCROOGE AND MARLEY']  ❌
修复后（改进版）:   ['SCROOGE', 'MARLEY']  ✅
```

**性能指标**：
```
NER 抽取:       ~0.2ms
模糊匹配:       ~2ms
实体链接准确率: > 95%
停用词覆盖率:   300+ 常见虚词/动词/噪声名词
```

### BERT Prototype 验证结果

我后来把这条链路补成了一个**实际可运行的 BERT Query NER 原型**，并在 `TGV-mini` 查询样例上做了快速验证。

**验证命令**：
```bash
/home/ycl1234/miniconda3/envs/graphrag/bin/python scripts/verify_bert_query_ner.py
```

**样例结果**：
```text
[1] LOT-20260401-A03 在 LASER-01 的 pulse_energy 漂移为什么会导致 hole_wall_roughness？
Predicted: ['LOT-20260401-A03', 'LASER-01', 'pulse_energy', 'hole_wall_roughness']

[2] 如何降低 PVD-04 在 barrier_deposition 后出现的 barrier_coverage_nonuniformity？
Predicted: ['PVD-04', 'barrier_deposition', 'barrier_coverage_nonuniformity']

[3] PLATE-01 的 bath_temperature 下降和 copper_fill_void 有什么关系？
Predicted: ['PLATE-01', 'TEM', 'copper_void', 'bath_temperature', 'copper_fill_void']

[4] CLEAN-02 的 rinse_conductivity 异常会不会引起 carryover_contamination？
Predicted: ['CLEAN-02', 'rinse_conductivity', 'carryover_contamination']
```

**当前阶段结论**：
- 4 条 TGV-mini query 的目标实体命中率：`13/13 = 100%`
- 对设备 ID、LOT ID、工艺参数、缺陷名这类结构化术语识别效果较稳
- 仍存在少量“补召回式”噪声候选，例如 `TEM`、`copper_void` 这类与 query 语义相关但非主目标实体

所以最准确的项目口径应该是：

- **当前已落地**：启发式 Query NER + Fuzzy Linking + HybridSearch 原型
- **当前已补充**：可跑的 BERT-assisted Query NER prototype
- **后续继续完善**：真正面向工业 query 的监督式 BERT NER、type-aware linking、intent-aware rerank

### 追问

- 为什么不用 LLM 做 NER
- 为什么当前代码先用启发式，而不是直接上 BERT
- BERT 版 Query NER 在整条链路里应该放在哪一层
- Fuzzy matching 的 threshold 怎么选
- 如果实体链接错了怎么办
- 联合解码的"联合"体现在哪里
- **新增的追问**：为什么需要停用词过滤

### 注意事项

- 强调"高频、低时延、稳定"是查询阶段的核心诉求
- 当前仓库落地的是轻量版；BERT 是为了把正式工业实现补齐到简历口径的演进方向
- "联合"指 NER + Linking + 检索的协同
- 主动提**停用词过滤**是解决整句误抽的关键

### 新增：关系问句为什么还要做 Structured Retrieval Plan？

如果 query 是：

```text
What is the relationship between Scrooge and Marley?
```

只做普通 Query NER + 子图扩展，系统往往会返回：

- 与 Scrooge 相关的背景
- 与 Marley 相关的背景
- 一堆主题中心节点

但这并不等于回答了“他们之间是什么关系”。

所以我后来把这类 query 单独语义化成：

```text
intent = relation_query
subject = SCROOGE
object = MARLEY
target = direct_relation | shortest_relation_path
```

然后检索阶段优先：

1. 找直接边
2. 找最短桥接路径
3. 最后才补背景实体

这样输出就不再是“背景知识堆叠”，而是直接对准关系本身。

**一句话八股原理**：

> 对关系问句，检索目标应该从“相关性最大”切换成“连接性最强”。

### 新增：为什么因果问句也要单独做 Structured Retrieval Plan？

因为因果问句问的不是：

- 哪些内容和这个问题相关

而是：

- 谁导致了谁
- 哪条因果链最强
- 哪些证据支持这个判断

所以像：

```text
Why did pulse_energy_drift lead to hole_wall_roughness?
```

内部要先语义化成：

```text
intent = causal_query
subject = pulse_energy_drift
object = hole_wall_roughness
target = root_cause | causal_chain | supporting_evidence
```

然后检索阶段优先：

1. 找因果词更强的路径
2. 找更贴近 effect 的路径
3. 再按 hop 和权重排序

最后 Narrator 也不能再只是列背景，而要优先输出：

- strongest causal chain
- supporting evidence
- related context

**一句话八股原理**：

> 对因果问句，系统要从“找相关信息”切换成“找解释链和证据链”。 

### 新增：为什么 causal query 还会答偏，应该怎么修？

当前 causal query 的一个典型现象是：

- planning 已经能得到 `intent=causal_query`
- 也能抽出 `subject` 和 `effect focus`
- 但最终 strongest path 仍可能被 `SPIRIT` 这类高连接度节点抢走

这说明问题不在 query planning，而在 execution。

我把根因拆成两层：

1. **BERT Query NER 缺少 domain gating**
   - 文学 query 里仍可能冒出 `AOI`、`seed_deposition` 这类工业术语
   - 说明候选被错误投影到了工业词表空间

2. **causal ranking 缺少 effect anchoring**
   - 虽然 plan 里已经有 `object=...`
   - 但如果排序阶段不强约束路径必须覆盖 effect 相关锚点
   - strongest path 还是会退化成“图上强相关”，而不是“问句对齐”

当前修法是：

- 在 `query_entity_extractor_bert.py` 中加入 query domain gating
- 在 `subgraph_mining.py` 中把 `rank_causal_paths(...)` 升级为：
  - subject 覆盖加分
  - effect anchor 覆盖加分
  - subject + effect 同时命中优先
  - 重复节点惩罚
- 在 `verify_bert_retrieval_bridge.py` 中把 causal seed 选择从“前两个 linked entities”改成“subject -> effect -> 其余实体”的顺序，保证 effect 相关节点真正进入子图扩展
- 在 `graph_narrator.py` 中把 causal answer 从 top-1 单路径输出，升级成“主路径 + Additional Supporting Causal Paths”的 top-k 共识式输出
- 在 `subgraph_mining.py` 中把节点层匹配从严格相等升级为包含式命中，并补了 `visit / ghost / warn / transformation` 这类更贴近文学因果语义的加权
- 在 `graph_narrator.py` 中继续补了 `causal answer synthesis`，会先从 top-k 路径抽 explanation fragments，再生成一条更自然的 why-answer
- 在 `subgraph_mining.py` 中新增了 `build_directed_causal_candidates(...)`，让 causal retrieval 不再完全按无向邻接扩图，而是优先沿 `source -> target` 方向走，并对逆向边降权
- 同时把 `prompts-tgv/extract_graph.txt` 收紧成更明确的定向关系类型：`CAUSES / LEADS_TO / RESULTS_IN / MITIGATES / PRECEDES`

最新回归结果也说明这个方向是有效的：

- 文学 query 不再误抽 `AOI / seed_deposition / via_offset`
- 关系问句 `What is the relationship between Scrooge and Marley?` 仍然能保持 direct edge 命中
- 因果问句虽然还没完全答到位，但 strongest path 已经从 `SPIRIT` 抢主链，进一步收敛到 `MARLEY / SCROOGE` 主轴附近，并且 `CHRISTMAS` 已经开始进入 Additional Supporting Causal Paths；Narrator 也会先给出一条 why-answer summary，而不是只展示路径
- 方向优先检索接入后，系统已经不再完全把 causal query 当无向邻接路径问题来做，这也更符合“因果图检索”的设计目标

**一句话八股原理**：

> causal query 的难点不只是“识别是因果问题”，而是让检索执行真正围绕 cause 和 effect 收敛，而不是被图里的高连接度节点带偏。

### 新增：现在的边是不是无向的，这和抽取阶段设计有没有关系？

更准确地说，不是“图里完全没有方向”，而是：

1. **存储层有 `source / target` 字段**
   - 所以图谱关系表不是完全无向边集

2. **但抽取层之前没有把因果方向建模得足够强**
   - 关系更多像“有关系”
   - 而不是严格的 `CAUSES / RESULTS_IN / MITIGATES / PRECEDES`

3. **检索层又为了提高召回，把边按无向邻接使用**
   - 这就让 causal retrieval 更像“相关路径搜索”
   - 不像“严格方向因果链搜索”

所以根因是两层叠加：

- 抽取层方向语义偏弱
- 检索层把弱方向进一步当成无向图处理

当前已经开始往这两层同时修：

**抽取层**
- 在 `prompts-tgv/extract_graph.txt` 中把关系类型收紧成更明确的定向类型：
  - `CAUSES`
  - `LEADS_TO`
  - `RESULTS_IN`
  - `MITIGATES`
  - `PRECEDES`
- 并且要求 `relationship_type` 真正输出进关系结构

**检索层**
- 在 `subgraph_mining.py` 中新增 `build_directed_causal_candidates(...)`
- causal retrieval 不再完全按无向邻接扩图，而是：
  - 优先沿 `source -> target` 方向走
  - 逆向边允许保召回，但会降权

**一句话八股原理**：

> 图谱有没有 `source/target` 字段，不等于系统已经在做强因果图检索；真正的因果图要同时依赖“强类型定向边 + 方向优先的候选路径生成”。

---

## 10. 子图挖掘与叙事化

### 问题

你说实现 2-3 hop 子图挖掘与 Top-K 路径排序，具体怎么做？

### 回答

**子图挖掘（2-hop DFS）**：

```python
def extract_causal_chains(seed_entities, relationships, max_hops=2, top_k_paths=10):
    # 构建邻接表
    adj = defaultdict(list)
    for rel in relationships:
        adj[rel.source].append(rel)
        adj[rel.target].append(rel)  # 无向图
    
    # DFS 搜索
    paths = []
    def dfs(current, path, depth):
        if depth >= max_hops:
            if len(path) > 0: paths.append(path[:])
            return
        for rel in adj.get(current, []):
            next_node = rel.target if rel.source == current else rel.source
            if rel not in path:
                path.append(rel)
                dfs(next_node, path, depth+1)
                path.pop()
    
    # 按权重排序
    paths.sort(key=lambda p: sum(r.weight or 0 for r in p), reverse=True)
    return paths[:top_k_paths]
```

**GraphNarrator 叙事化**：

```python
class GraphNarrator:
    def narrate_subgraph(self, entities, relationships):
        lines = ["## Relevant Knowledge\n", "### Key Entities"]
        for e in entities:
            lines.append(f"- **{e.title}** ({e.type}): {e.description[:200]}...")
        lines.append("\n### Relationships and Connections")
        for rel in relationships:
            lines.append(f"- {rel.source} is related to {rel.target}: {rel.description[:100]}")
        return "\n".join(lines)
```

**性能指标**：
```
子图挖掘:     ~5ms
叙事化:       ~0.1ms
上下文扩充:   +847% 平均 (最高 +1705%)
```

### 追问

- 为什么是 2-3 hop，不是更多
- 路径排序的权重怎么来
- 子图过大怎么控噪声
- 叙事化模板怎么设计

### 注意事项

- 强调"围绕问题构建局部证据网络"，不是暴力喂整图
- +847% 和 +1705% 是核心数字，要背熟

---

## 11. HybridSearch 混合检索引擎

### 问题

你简历里写了 HybridSearch，具体架构是什么？

### 回答

**架构设计**：

```
Query → Query NER → 实体链接 → 语义检索
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        Global Search      Local Search
        (宏观主题定位)     (精细子图检索)
              ↓                 ↓
              └────────┬────────┘
                       ↓
                  上下文融合
                       ↓
                  截断排序
                       ↓
                  大模型生成
```

**四阶段检索**：

| 阶段 | 模式 | 适用场景 | 示例 |
|-----|------|---------|------|
| Global | 社区摘要 | 宏观总结 | 最常见的工艺-缺陷关联 |
| Local | 子图召回 | 局部路径 | 孔径过小和哪些参数相关 |
| Drift | 探索扩展 | 模糊问题 | 质量波动大，不确定哪道工序 |
| Basic | 向量检索 | 事实查询 | 某缺陷的定义 |

**融合策略**：
- Global 结果用于主题定位
- Local 结果用于证据细节
- 按优先级和相似度排序
- 上下文长度超过阈值时截断

### 追问

- 四种 search 怎么选择
- 上下文融合怎么排序
- 截断策略是什么
- 如果 Global 和 Local 结果冲突怎么办

### 注意事项

- 强调"Global→Local→Fusion"的多轮推理 Pipeline
- 不要讲成四种 search 是互斥的，它们是工具箱

---

## 12. 技术原理深度解析

### 12.1 Query NER 联合解码机制原理

### 问题

为什么查询阶段不用 LLM 做实体抽取，而是设计轻量级的 Query NER + Fuzzy Linking？

### 回答

核心原因是**查询阶段和索引阶段的诉求不同**。

**索引阶段**：面对开放文档，实体类型不确定，需要灵活抽取，适合用 LLM。

**查询阶段**：
1. **高频低时延**：查询可能是秒级并发，LLM 50-200ms 的延迟 unacceptable
2. **实体类型可控**：工艺场景里的实体类型是有限的（缺陷类型、设备编号、工序名）
3. **稳定性要求高**：规则不受模型波动影响

**为什么需要停用词过滤？**

原始正则 `\b[A-Z][A-Z\s]{1,50}\b` 的问题是：
```
查询: "What is the relationship between Scrooge and Marley?"
误抽: "WHAT IS THE RELATIONSHIP BETWEEN SCROOGE AND MARLEY"  (整句被匹配)
```

自然语言查询中 **60%+ 是虚词**（the/is/are/and/relationship），必须通过停用词表过滤，否则正则过于贪婪会匹配整句。

**联合解码的核心思想**：
- **精确匹配（NER）**：~0.2ms，召回候选实体
- **停用词过滤**：300+ 常见虚词/动词/噪声名词，避免误抽
- **模糊匹配（Fuzzy）**：~2ms，处理别名、缩写、大小写变体
- **优先级融合**：精确匹配结果优先级高于语义检索

**八股原理总结**：

| 设计决策 | 原理 | 解决的问题 |
|---------|------|-----------|
| 启发式而非 LLM | 查询阶段强调高频、低时延 | 时延从 50ms+ 降至 ~0.2ms |
| 停用词过滤 | 自然语言中 60%+ 是虚词 | 解决整句误抽问题 |
| 长度限制（3-30字符） | 单实体通常 2-4 词 | 避免正则贪婪匹配 |
| Fuzzy Linking（0.8） | 编辑距离对字符变体敏感 | 处理别名/缩写/大小写变体 |
| 精确匹配优先 | NER 结果是确定性信息 | 确保明确提到的实体一定被包含 |

这种设计在保证召回率的同时，将时延控制在 ~3ms，满足高频查询需求。

### 追问

- Fuzzy Matching 的 threshold 怎么选
- 如果 NER 漏了实体怎么办
- 这种方法的局限性是什么
- **为什么正则 \b[A-Z][A-Z\s]{1,50}\b 会匹配整句**

### 注意事项

- 强调"查询阶段"和"索引阶段"的差异
- threshold=0.8 是经验值，可解释
- 主动解释**停用词过滤的必要性**（正则贪婪匹配问题）

---

### 12.2 子图挖掘算法原理

### 问题

子图挖掘为什么用 DFS 而不是 BFS？为什么是 2-3 hop 而不是更深？

### 回答

**DFS vs BFS 的选择**：

工艺追因的核心诉求是找到**完整的因果链**，而非所有直接关联。

- **DFS（深度优先）**：优先探索一条路径到底，适合找"缺陷→原因→措施"链条
- **BFS（广度优先）**：优先找到所有邻居，适合概览分析

追因场景需要完整的因果路径，DFS 更符合业务逻辑。

**2-3 hop 的权衡**：

| 跳数 | 覆盖范围 | 噪声 | 适用场景 |
|-----|---------|------|---------|
| 1-hop | 直接关联 | 低 | 简单查询 |
| 2-hop | 缺陷→参数→工序 | 中 | 大多数追因场景 |
| 3-hop | 跨工序传递 | 较高 | 复杂根因分析 |
| >3-hop | 全局 | 极高 | 不适用 |

2-hop 覆盖了"缺陷→原因→措施"或"缺陷→参数→工序"的完整链条，是精度和召回的最佳平衡点。

**路径排序原理**：
按关系权重排序而非路径长度，因为工业知识中"导致"和"相关"的置信度差异很大。

### 追问

- 如果 2-hop 找不到答案怎么办
- 环路怎么处理
- 权重怎么计算

### 注意事项

- 强调"业务诉求决定技术选型"
- DFS/BFS 不是对错，是适合不适合

---

### 12.3 GraphNarrator 叙事化原理

### 问题

为什么要做 GraphNarrator 叙事化？直接把图谱数据喂给 LLM 不行吗？

### 回答

直接把图谱数据（节点+边）喂给 LLM 存在三个问题：

1. **理解成本高**：LLM 对自然语言的理解远强于结构化数据
2. **上下文冗长**：原始图谱格式包含大量元数据噪声（ID、类型标记等）
3. **关系不明显**：孤立的三元组难以体现因果逻辑

**叙事化的核心思想**：
用模板将结构化数据转换为自然语言，既保留结构化信息，又符合 LLM 输入习惯。

**模板设计原则**：
1. **实体优先**：先介绍关键实体，建立认知锚点
2. **关系递进**：按因果链顺序组织
3. **属性精简**：只保留最关键的 2-3 个属性
4. **证据溯源**：标注数据来源，支持可解释性

**效果**：上下文扩充率 +847%，同时保持 LLM 回答质量。

### 追问

- 模板会不会限制灵活性
- 如果子图很大怎么办
- 怎么验证叙事化效果

### 注意事项

- 强调"结构化→自然语言"是桥接层，不是替代图谱
- 模板是可控性和灵活性的权衡

---

### 12.4 HybridSearch 混合检索原理

### 问题

为什么要设计 HybridSearch？单一检索模式有什么问题？

### 回答

单一检索模式各有局限：

| 模式 | 优势 | 劣势 |
|-----|------|------|
| 纯向量检索 | 语义泛化强 | 无法捕捉结构化关系 |
| 纯图检索 | 关系推理强 | 缺乏语义泛化能力 |
| 纯关键词 | 精确 | 召回率低 |

**HybridSearch 的设计哲学**：
借鉴人类"由粗到细"的认知模式：
1. **Global（粗）**：社区摘要定位宏观主题
2. **Local（细）**：子图挖掘深入具体证据
3. **融合（合）**：综合形成完整结论

**融合策略的核心**：
- **精确匹配优先**：Query NER 识别的实体优先级最高（确定性）
- **语义匹配补充**：向量检索召回相关但不完全匹配的信息（相关性）
- **全局视角平衡**：社区摘要提供背景知识，防止局部偏差

这种设计在精度、召回、时延之间取得平衡。

### 追问

- 如果 Global 和 Local 结果冲突怎么办
- 权重怎么调
- 什么时候只用 Local 不用 Global

### 注意事项

- 强调四种 search 是工具箱，不是互斥
- Global→Local→Fusion 是 Pipeline 思想

---

### 12.5 vLLM 推理优化原理

### 问题

为什么要从 Ollama 迁移到 vLLM？优化的核心原理是什么？

### 回答

**迁移的核心原因**：索引阶段是**大批量、长时间、高并发**场景，Ollama 在这种场景下有瓶颈。

**Ollama 的局限**：
- 缺乏 batching：每个请求独立处理，GPU 利用率低
- 缓存管理弱：长时间运行后 KV Cache 碎片化
- 超时限制：不适合持续数小时的索引任务

**vLLM 的核心优化原理**：

1. **Continuous Batching**：
   - 动态将多个请求 batch 在一起
   - 不同长度请求智能 padding
   - 吞吐提升 38.6%

2. **PagedAttention**：
   - 将 KV Cache 分页管理
   - 减少显存碎片
   - 支持更大 batch size

3. **投机解码（Speculative Decoding）**：
   - 用 7B draft model 预测 token
   - 14B main model 验证
   - 减少实际 forward 次数

**效果**：吞吐从 300.6 tok/s 提升到 416.6 tok/s，长时稳定性显著改善。

### 追问

- 投机解码为什么能加速
- 7B 和 14B 词表必须匹配吗
- 为什么还要保留 Ollama

### 追问：为什么还要保留 Ollama

**回答**

我在这条实验线里并不是把所有服务一次性都迁到 vLLM，而是先做**生成侧对比**，把变量控制住。

具体做法是：

- completion/query：迁到 vLLM
- embedding：先保留在 Ollama

这样做有两个原因：

1. **先控制变量**
   - 如果一开始把 completion 和 embedding 两层都一起替换，后面性能变化就很难判断到底来自哪一层
   - 先只替换生成侧，可以更清楚比较 Ollama 和 vLLM 在高并发抽取阶段的吞吐和稳定性差异

2. **embedding 在这套链路里本来就是独立故障点**
   - 我后面发现，`bge-m3` 在 Ollama `/v1/embeddings` 路径上会出现 `NaN/500`
   - 所以生成侧迁到 vLLM 后，主链路已经明显改善，但最后仍可能被 embedding 后端拖住

后续我再把 embedding 模型替换成 `mxbai-embed-large`，才把整条链路收尾跑通。

所以保留 Ollama 不是因为 vLLM 不行，而是：

- 前期为了控制变量
- 后期为了把生成优化和 embedding 稳定性拆开分析

### 注意事项

- 强调"从能跑到可持续高吞吐"
- 词表匹配是投机解码的关键前提

---

## 13. 项目评测与验证

### 问题

你说构建了完整的评测体系，具体评测了什么？

### 回答

**四维度评测**：

| 维度 | 指标 | 方法 |
|-----|------|------|
| 生成质量 | 实体准确率、关系正确性 | 人工抽检 + 规则校验 |
| 检索性能 | QPS、P50/P95 延迟 | 自动化压力测试 |
| 上下文质量 | 扩充率、信息增益 | 字符统计 + 人工评估 |
| 端到端效果 | 相关性、可解释性 | 业务场景测试 |

**具体结果**：

1. **生成质量**：
   - 实体抽取准确率：94.4%（人工抽检 100 份文档）
   - 关系正确率：87.3%（抽检 200 条关系）

2. **检索性能**（压力测试）：
   ```
   NER 抽取:      71.32 QPS,  P50=51.87ms
   子图挖掘:      121.36 QPS, P50=10.98ms
   完整 Pipeline: 41.89 QPS,  P50=67.16ms, 成功率 100%
   ```

3. **上下文质量**：
   ```
   平均扩充率:     +847%
   最佳案例:       +1705% (Scrooge & Marley 查询)
   ```

4. **端到端对比**（Baseline vs GraphRAG）：
   ```
   回答相关性:     3.2 → 4.5 (+40.6%)
   证据可溯源:     1.8 → 4.2 (+133%)
   建议可操作性:   2.5 → 4.0 (+60%)
   ```

### 追问

- 评测数据怎么来的
- Baseline 怎么定义的
- 人工评估的样本量多少
- 有没有做过消融实验

### 注意事项

- 数字要准确，不要编造
- 强调"precision 优先"的工业场景特点
- 可以提"人工抽检 + 自动化测试"的组合方法

---

## 14. 知识图谱的实体和关系怎么设计

### 问题

你们图谱里主要有哪些实体和关系？

### 回答

实体设计是从业务问题反推的，覆盖完整的工艺追因链。

**核心实体**：

```
产品层: ProductLot (批次) → Board (板) → Hole (孔)
工艺层: ProcessStep (工序) → Recipe (配方) → ProcessParameter (参数)
设备层: Tool (设备) → Material (材料/试剂)
缺陷层: DefectType (缺陷类型) → DefectCause (缺陷原因) → ActionMeasure (措施)
知识层: DetectionEvent (检测事件) → ExpertRule (专家规则) → Document (文档)
```

**核心关系**：

```
层级关系: LOT_HAS_BOARD → BOARD_HAS_HOLE
工艺关系: STEP_USES_TOOL, STEP_USES_MATERIAL, STEP_HAS_PARAMETER
事件关系: EVENT_OCCURS_AT_STEP, EVENT_ON_BOARD, EVENT_ON_HOLE
缺陷关系: DETECTS_DEFECT, DEFECT_HAS_CAUSE, CAUSE_HAS_MEASURE
因果关联: DEFECT_RELATED_TO_PARAMETER, DEFECT_RELATED_TO_STEP
证据关联: DOCUMENT_SUPPORTS_RELATION, INFERENCE_REFERENCES_ENTITY
```

### 追问

- 为什么不把更多字段都建成实体
- 为什么事件也要建模
- 关系里怎么表达时间和条件
- 5 万条知识库是什么意思

### 注意事项

- 强调 schema 不是越复杂越好，而是要服务后续检索和推理
- 回答时尽量用"缺陷 -> 参数 -> 工序 -> 原因 -> 措施"的链来讲
- 5 万条指的是结构化事实单元（实体+关系+事件），不是单纯节点数

---

## 15. 为什么不只用三元组

### 问题

你简历里写了三元组/四元组，为什么要做四元组？

### 回答

因为工业知识里很多关系不是无条件成立的，而是在特定工序、参数窗口或检测条件下成立。三元组只能表达"谁和谁有关"，但不能表达"在什么条件下有关"。

所以我会在三元组基础上增加条件上下文，例如：

```
三元组: 高激光功率密度 → may_cause → 孔壁重铸层增厚
条件:   薄玻璃基板、扫描速度 > 500mm/s
证据:   《工艺分析报告-2024-Q2》第 15 页
```

这样可以减少错误泛化，更符合工业因果知识的表达方式。

### 追问

- 四元组里你保留的条件通常有哪些
- 为什么不把条件做成属性而是单独保留
- 条件怎么用于后续推理

### 注意事项

- 不要把四元组讲得过于理论化
- 重点突出"减少错误泛化"和"保留约束条件"

---

## 16. 抽取 Prompt 怎么设计

### 问题

你说自定义了抽取 Prompt，核心思路是什么？

### 回答

核心原则是**"宁可少抽，不要瞎抽"**。

我会在 Prompt 里强调：

1. **保守抽取**：只能抽原文明确支持的实体和关系
2. **不确定性处理**：不确定就返回空，不要补常识
3. **条件约束**：关系需要保留条件上下文
4. **Schema 校验**：输出必须满足预定义结构
5. **证据绑定**：尽可能标注来源文档和段落

工业场景里最怕的是看起来合理但其实没有依据的因果关系，所以抽取阶段宁可保守。

### 追问

- 如何减少 hallucination extraction
- 你怎么做 schema 校验
- prompt 失效时怎么排查
- 保守抽取会不会漏掉重要信息

### 注意事项

- 不要把功劳都归给 prompt，记得补充"还有 schema 校验和人工抽检"
- 承认"precision 优先，recall 可能牺牲"是设计权衡

---

## 17. 如何验证抽取质量

### 问题

知识抽取质量怎么验证？

### 回答

我主要做三层控制。

**第一层：格式和 schema 合法性检查**
- 实体类型是否在预定义列表
- 关系类型是否合法
- 必填字段是否完整

**第二层：语义正确性抽检**
- 从不同来源文档抽样
- 检查实体边界、关系方向
- 验证条件是否保留、是否存在幻觉抽取
- 样本量：100 份文档片段，200 条关系

**第三层：业务结果验证**
- 看最终检索和追因场景里，知识是否真的能支持问答
- 端到端测试：50 个典型查询，人工评估回答质量

工业场景里我更重视 **precision**（92.4%），因为错一条因果关系比漏一条关系更危险。

### 追问

- 你们有没有人工校验
- 为什么更看重 precision 而不是 recall
- 你们有没有做过 error case 分类
- 如果图谱有噪声怎么办

### 注意事项

- 工业场景回答"precision 优先"会比较加分
- 如果没有完整指标，重点讲方法，不硬编数字

---

## 18. 为什么查询阶段还要用轻量实体抽取

### 问题

为什么不用 LLM 直接完成查询理解，还要加一个轻量实体抽取？

### 回答

因为**查询阶段强调的是高频、稳定、低时延**。

索引阶段面对的是开放文档，适合用 LLM 做灵活抽取；但查询阶段的实体类型比较可控（缺陷类型、工艺参数、工艺步骤、设备、批次号等）。

所以我用**启发式规则 + 停用词过滤**先把问题里的关键实体抽出来（**~0.2ms**），构建查询锚点，再围绕这些锚点去做子图召回和证据扩展。这样可以：
- 提高稳定性（规则不受模型波动影响）
- 更容易控制时延（LLM 需要 50-200ms）
- 减少 LLM 调用次数和成本

**为什么必须是"启发式 + 停用词过滤"？**

早期版本用简单正则 `\b[A-Z][A-Z\s]{1,50}\b`，结果：
```
查询: "What is the relationship between Scrooge and Marley?"
误抽: "WHAT IS THE RELATIONSHIP BETWEEN SCROOGE AND MARLEY"  (整句匹配)
```

根本原因是自然语言查询中 **60%+ 是虚词**（the/is/are/and/relationship），必须通过 300+ 停用词表过滤，否则正则过于贪婪。

**联合解码**：NER 结果与向量检索结果按优先级融合，NER 的精确匹配优先级高于语义检索的相似匹配。

### 追问

- 联合解码具体是什么意思
- 为什么不用 BERT 而用启发式规则
- 查询阶段纯用 LLM 会有什么问题
- 如果 NER 没抽出来怎么办
- **停用词过滤具体过滤了哪些词**

### 注意事项

- 不要把轻量抽取讲成替代 LLM，它是查询链路里的前置锚点识别模块
- 核心关键词是"稳定、低时延、高频"
- 主动解释**停用词过滤的必要性**和**300+ 停用词表的设计**

---

## 19. 为什么要把结构化图转成自然语言

### 问题

既然已经有图谱，为什么还要把结构化数据转成自然语言再喂给 LLM？

### 回答

因为大模型虽然能处理结构化信息，但直接给节点和边，理解成本高，而且输出不稳定。把子图按模板转成自然语言证据段，可以降低图谱数据和大模型之间的理解障碍。

例如：

**结构化形式**：
```
节点: [孔壁粗糙, 激光功率密度, 湿法刻蚀]
边: [孔壁粗糙-相关参数->激光功率密度, 孔壁粗糙-相关工序->湿法刻蚀]
```

**叙事化形式**：
> "历史上孔壁粗糙通常出现在湿法刻蚀后，且常与刻蚀时长偏短、刻蚀液浓度偏低相关。建议优先排查刻蚀工序参数。"

这种形式更利于模型做后续推理、比较和总结。

### 追问

- 模板怎么设计
- 模板会不会损失结构信息
- 为什么不直接做图神经网络推理
- 如果子图很大，叙事化会不会超长

### 注意事项

- 强调这是"结构化信息到自然语言上下文"的桥接层
- 不要说成图谱没用，而是图谱负责组织，自然语言负责表达

---

## 19A. 查询理解与模型基础追问

### 问题 1

大模型做实体抽取会导致搜索词丢失额外信息、信息熵减少，这个问题怎么解决？

### 回答

这个问题的本质是：**如果把原始 query 直接压缩成几个实体名，虽然方便检索，但会损失 query 里的约束、语气、目标和比较关系。**

例如用户问：

> `What is the relationship between Scrooge and Marley?`

如果只留下：

- `SCROOGE`
- `MARLEY`

那就丢了下面这些信息：

- 用户问的是 **relationship**
- 不是单实体介绍
- 不是主题概览
- 也不是开放式总结

这就是“信息熵减少”的工程表现：**query 从一个带任务意图的语义表达，被压缩成了几个离散锚点。**

所以正确做法不是“让实体抽取替代 query”，而是把它作为**查询理解的一部分**。我通常会把 query 拆成两条并行信号：

1. **实体锚点信号**
   - 用轻量 Query NER 抽出显式实体
   - 作用是保证关键对象不漏

2. **语义任务信号**
   - 保留原始 query
   - 用来识别这是“关系问答 / 因果问答 / 比较问答 / 建议问答”
   - 指导后续的子图召回、排序和模板输出

也就是说：

- **实体抽取解决 recall**
- **原始 query 解决 intent 和约束保留**

在你当前代码里，落地的是第一部分：

- `QueryEntityExtractor.extract(query)`
- `get_entity_by_name_fuzzy(...)`

对应文件：
- [query_entity_extractor.py](/mnt/data2/ycl/graphrag/scripts/custom_modules/query_entity_extractor.py)

如果后续继续加强，我建议在这之上再补一层：

- query intent classification
- relation-aware ranking
- constraint extraction（如时间、比较对象、因果方向）

**八股原理总结**：

| 设计问题 | 原理 | 工程结论 |
|---|---|---|
| 为什么只做实体抽取不够 | 实体是 query 的一部分，不是 query 全部语义 | 实体锚点和原始语义必须并行保留 |
| 为什么会“信息熵减少” | 任务意图、比较关系、因果方向在实体压缩时会丢失 | 不能把 NER 输出当成 query 的替身 |
| 应该怎么做 | 显式实体 + 原始 query + 检索融合 | 两路信息同时参与召回和排序 |

### 追问

- 如果 query 里没有明确实体怎么办
- 如果 query 是比较型问题怎么办
- 如果 query 包含条件约束（时间、设备、批次）怎么办

### 问题 2

如何将 query 语义化？

### 回答

“query 语义化”不是把一句话改写得更漂亮，而是把原始问题拆成几个可计算的语义信号，供后面的检索和推理模块使用。

我建议把 query 语义化拆成 4 层：

1. **实体层**
   - query 里提到了哪些核心对象
   - 如设备、缺陷、工序、参数、人物名

2. **意图层**
   - 用户是在问：
     - 定义
     - 关系
     - 原因
     - 建议
     - 对比
     - 趋势

3. **约束层**
   - 是否有时间、批次、工序阶段、设备、空间范围等限制

4. **输出层**
   - 应该返回：
     - 事实答案
     - 因果链
     - 证据列表
     - 建议项

拿你当前项目举例，query 语义化最自然的流程是：

```text
原始 Query
→ Query NER 抽实体
→ Fuzzy Linking 对齐知识库实体
→ 识别 query 类型（关系 / 因果 / 比较 / 建议）
→ 根据类型决定检索策略
   - relation: 双实体桥接路径优先
   - causal: 2-3 hop 因果链优先
   - suggestion: 缺陷→原因→措施链优先
→ 再做 Global/Local/Hybrid 检索
```

你现在仓库里已经落地的是前半段：

- Query NER
- Fuzzy Linking
- 子图挖掘
- GraphNarrator

对应文档位置：
- [tgvgraphrag.md](/mnt/data2/ycl/graphrag/doc/tgvgraphrag.md#L739)

如果你面试被问“query 语义化具体怎么做”，一个很稳的回答是：

> 我不会直接把 query 当字符串丢给向量检索，而是先做语义拆解：先抽实体锚点，再识别 query 是关系、因果还是建议类问题，然后按问题类型决定后面的子图扩展和上下文构造方式。

**八股原理总结**：

| 语义化层次 | 作用 | 典型方法 |
|---|---|---|
| 实体层 | 把 query 锚到知识库节点 | Query NER + Linking |
| 意图层 | 决定检索模式 | 规则 / 分类器 / prompt 路由 |
| 约束层 | 保留时间、批次、工序等限制 | regex / parser / slot filling |
| 输出层 | 决定回答模板 | Local / Global / Hybrid + Narrator |

### 追问

- 如何识别关系型 query 和因果型 query
- 如何把约束条件传递给检索器
- 语义化和 query rewrite 的区别是什么

### 问题 3

说一下 BERT 结构，BERT 对中文可不可以做字和词的区分？

### 回答

BERT 的核心结构可以概括成：

1. **输入层**
   - Token Embedding
   - Position Embedding
   - Segment Embedding

2. **编码层**
   - 多层 Transformer Encoder
   - 每层都包含：
     - Multi-Head Self-Attention
     - Feed Forward Network
     - Residual Connection
     - LayerNorm

3. **预训练目标**
   - MLM（Masked Language Modeling）
   - NSP（原始 BERT 里有，很多后续中文模型会弱化或替换）

所以它本质上是一个：

> 基于双向 self-attention 的上下文编码器

**BERT 对中文能不能区分字和词？**

能，但要把“区分”说准确：

- 原生中文 BERT 通常主要以**字 / 子词粒度**做 tokenization
- 它不天然像传统分词器那样先切出稳定词边界
- 但通过上下文 self-attention，它可以在表示空间里学到“哪些字组合成词、短语、实体”

换句话说：

- **输入上**更偏字或子词
- **表示上**可以隐式建模词级语义

所以如果面试官问：

> 中文 BERT 能不能区分字和词？

一个准确回答是：

> 可以做，但不是靠显式词边界，而是通过字/子词级输入 + 上下文建模，在隐空间里学到词级组合关系。如果业务特别依赖词边界，可以再叠加中文分词、词典特征或 lexicon-aware 模型。

**结合你当前项目的口径**

当前仓库里的 Query NER 不是 BERT 版，而是启发式版。  
如果你以后真要做 BERT Query NER，更合理的落地方向是：

- 输入：中文 query 字/子词序列
- 输出：BIO/BIESO 序列标注
- 后处理：实体 span -> alias normalization -> fuzzy linking

**八股原理总结**：

| 点 | 结论 |
|---|---|
| BERT 结构 | 输入嵌入 + 多层 Transformer Encoder + MLM 预训练 |
| 中文输入粒度 | 更偏字 / 子词，而不是强依赖显式分词 |
| 能不能区分词 | 能在表示空间中隐式建模词级语义 |
| 什么时候要额外做词级增强 | 实体边界敏感、术语词典强依赖、行业缩写多的场景 |

### 追问

- 中文 BERT 为什么不一定先做分词
- BERT 和 BiLSTM-CRF 做 NER 的差别是什么
- WordPiece / SentencePiece 对中文有什么影响

### 问题 4

说一下 LN 和 BN，以及分别适合什么场景。

### 回答

BN 和 LN 都是归一化方法，但归一化维度不同，因此适用场景也不同。

## BatchNorm（BN）

BN 是在 **batch 维度** 上做归一化。  
它用一个 batch 内的均值和方差来标准化当前层输出。

特点：

- 依赖 batch statistics
- 对 batch size 比较敏感
- 在 CNN / 视觉任务里非常常见
- 对大 batch 训练比较友好

优点：

- 训练更稳定
- 收敛更快
- 对卷积网络尤其有效

缺点：

- 小 batch 效果容易变差
- 在线推理和训练统计不完全一致
- 在序列模型、变长输入、batch 很小的场景不够稳

## LayerNorm（LN）

LN 是对 **单个样本内部的特征维度** 做归一化。  
不依赖同 batch 其他样本。

特点：

- 不依赖 batch size
- 对变长序列更稳定
- 特别适合 Transformer / NLP

优点：

- 训练和推理行为更一致
- 很适合小 batch 或 batch size 波动大的场景
- 更适合文本、序列、在线服务

缺点：

- 在某些 CNN 场景下不如 BN 高效
- 不能直接享受“大 batch 统计稳定”的好处

## 分别适合什么场景

| 方法 | 更适合的场景 |
|---|---|
| BN | CNN、图像、大 batch 训练、离线训练稳定场景 |
| LN | Transformer、NLP、变长序列、小 batch、在线推理场景 |

**和你项目的关系**

你现在用到的主生成模型和检索增强链路，本质上都是 Transformer 路线，所以背后的主流归一化是 **LayerNorm**，不是 BatchNorm。

**八股原理总结**：

| 问题 | BN | LN |
|---|---|---|
| 归一化维度 | batch 维 | 特征维 |
| 是否依赖 batch size | 是 | 否 |
| 推理和训练一致性 | 较弱 | 更强 |
| 典型场景 | CNN / CV | Transformer / NLP |

### 追问

- 为什么 Transformer 普遍用 LN 不用 BN
- 小 batch 时 BN 为什么会不稳定
- RMSNorm 和 LN 的区别是什么

### 问题 5

BN 的离线 / 在线不一致问题怎么解决？

### 回答

这个问题本质上来自：

- **训练时**：BN 用当前 mini-batch 的均值和方差
- **推理时**：BN 用训练阶段累积的 running mean / running var

如果训练分布和推理分布不一致，或者 batch 太小，BN 就可能出现“训练好好的，线上效果掉”的问题。

常见解决方式有 5 类：

1. **增大 batch size**
   - 让训练统计更稳定
   - 这是最直接但不一定最现实的方法

2. **使用更稳的统计方式**
   - 比如 SyncBN
   - 在多卡训练时同步统计量

3. **训练后重新校准 BN 统计量**
   - 用一批接近线上分布的数据重新跑 forward
   - 更新 running mean / var

4. **冻结 BN**
   - finetune 或小数据场景里，直接冻结 BN 参数和统计量

5. **直接换归一化方案**
   - 在小 batch、在线推理、序列建模里，很多时候直接换成 LN / GN 会更稳

如果面试官追问“工程上你会怎么选”，一个很稳的回答是：

> 如果是 CNN 大 batch 训练，我优先保留 BN，并通过足够 batch size、SyncBN 或统计量重校准解决一致性问题；如果是小 batch、在线推理、NLP/Transformer 场景，我通常会优先使用 LN，而不是去强行修 BN。

**八股原理总结**：

| 问题来源 | 解决思路 |
|---|---|
| 训练时用 batch 统计，推理时用 running 统计 | 重新校准、冻结或同步统计 |
| batch 太小，统计不稳定 | 增大 batch、用 SyncBN、换 LN/GN |
| 数据分布漂移 | 用更接近线上分布的数据做 BN calibration |

### 追问

- SyncBN 的原理是什么
- 为什么小 batch 会让 BN 失效
- 什么时候应该直接放弃 BN 改用 LN 或 GN

---

## 20. 为什么原生推理服务会出问题

### 问题

你简历里写 Ollama 在 Map-Reduce 抽取阶段出现 KV Cache 显存碎片化和 10 分钟后超时，你是怎么发现和分析的？

### 回答

我不是先主观认定是 KV Cache，而是先从**现象倒推**：

1. **单次短请求正常**：简单问答没问题
2. **长时间运行后出问题**：大批量图谱抽取场景下，服务开始还能跑
3. **延迟持续上升**：随着时间拉长，请求延迟持续上升，尾部请求越来越慢
4. **约 10 分钟超时**：最后出现 HTTP 超时中断
5. **显存表象**：看起来还有余量，但服务越来越慢，不像简单 OOM

**定位结论**：这更像缓存管理和调度效率下降，而不是显存不足。

**验证**：切换到 vLLM（PagedAttention + 更好的调度）后问题明显缓解，确认瓶颈主要在推理服务层。

### 追问

- 为什么抽取任务比聊天更容易触发这个问题
- 你做过哪些监控和日志分析
- 为什么不是网络层问题
- 你是怎么区分"显存不足"和"显存碎片"的

### 注意事项

- 不要把自己说成做了底层源码调试，除非真改过
- 用"根据现象定位到推理服务层"这种表述更稳

---

## 21. 为什么换到 vLLM 效果更好

### 问题

你为什么从 Ollama 切到 vLLM，本质收益是什么？

### 回答

本质上不是简单换了个框架，而是把推理服务从"能跑"升级成了**更适合批量抽取任务的服务化形态**。

图谱索引场景的特点是：
- 请求数量多（几百到几千个 chunk）
- 文本长度差异大（几百到几千 token）
- 运行时间长（数十分钟到数小时）

所以更看重：
- **KV cache 管理能力**：PagedAttention 的显存池化管理
- **Continuous Batching**：不同长度请求混合调度
- **整体吞吐稳定性**：长时间高并发不 degrading

**量化收益**：吞吐从 300.634 tok/s 提升到 416.637 tok/s（+38.6%）。

### 追问

- 你说的收益具体体现在哪些指标上
- 你是否实际改了 vLLM 源码
- Speculative Decoding 在你场景里起了什么作用
- 多卡隔离怎么做的

### 注意事项

- 如果没改源码，不要说"我实现了底层机制"
- 可以说"利用其推理服务机制和调度能力提升了吞吐和稳定性"

---

## 22. 你做了哪些稳定性优化

### 问题

除了换推理后端，你还做了哪些工程优化？缓存和并发控制的设计原理是什么？

### 回答

我做的不是单点优化，而是一组**稳定性策略组合**，核心思想是"渐进加载、容错重试、状态留存"。

**并发控制原理**：

| 策略 | 参数 | 原理 |
|-----|------|------|
| 渐进加载 | `stagger=0.3s` | 避免瞬间打满服务，给系统缓冲时间 |
| 队列限制 | `max_num_seqs=4` | 控制并发请求数，防止队列堆积 |
| 超时控制 | `timeout=600s` | 长尾请求不阻塞整体流程 |
| 指数退避 | `base=1s, max=60s` | 失败时逐步增加等待，避免雪崩 |

**缓存策略原理**：

索引阶段是长时任务，5万条知识可能需要数小时。缓存设计目标：
1. **断点续传**：中途失败可从断点恢复，而非重新开始
2. **快速迭代**：修改 prompt 后只需重新计算变化部分
3. **成本控制**：减少重复调用 LLM 的 API 费用

缓存粒度设计：
- **实体缓存**：chunk → 实体列表（24h TTL）
- **关系缓存**：实体对 → 关系（24h TTL）
- **向量缓存**：文本 → embedding（48h TTL）
- **社区缓存**：图 → 社区划分（12h TTL）

**稳定性组合拳**：

| 策略 | 具体做法 | 作用 |
|-----|---------|------|
| 并发限制 | `max_num_seqs=4` | 避免把模型服务瞬间打满 |
| 超时控制 | 请求级 + 任务级双重超时 | 避免长尾请求无限阻塞 |
| 指数退避重试 | 初始 1s，最大 60s，最多 3 次 | 解决短时抖动问题 |
| 缓存复用 | 启用 GraphRAG 文件缓存 | 断点续传，减少重复抽取 |
| 日志留痕 | 详细记录每个 chunk 处理状态 | 失败样本回放，便于排查 |
| 健康检查 | 启动前确认模型服务可用 | 减少无效请求 |

工业项目里重要的不只是最快，而是**能稳定跑完**。

### 追问

- 并发为什么不尽量拉高
- 重试会不会放大队列堆积
- 你们如何区分临时失败和系统性失败
- 如果某个 chunk 一直失败怎么办

### 注意事项

- 回答时多强调"稳定跑完"和"可排查性"
- 可以提"断点续传"是长时任务的关键

---

## 23. 图像在这个系统里扮演什么角色

### 问题

这个项目和缺陷图像是什么关系？GraphRAG 会直接看图吗？

### 回答

GraphRAG **不直接替代视觉检测模型**。图像层面的检测、分割、定位、分类通常先由外部视觉模型（如 YOLO）完成，GraphRAG 负责把视觉输出和文档知识、工艺事件、设备日志、专家规则连接起来。

因此图像在系统里主要以两类信息存在：

1. **图像元数据**：路径、采集时间、板号、孔号、相机信息
2. **视觉结果**：缺陷类型、置信度、检测框、分割区域、人工复判结果

这些结构化输出作为图谱实体（DetectionEvent、ImageAsset）的一部分接入系统，支撑后续的追因分析。

### 追问

- 为什么不直接上多模态大模型
- 视觉结果错误会怎么影响图谱
- 未来怎么扩展到多模态问答
- 图像怎么和文本知识关联

### 注意事项

- 划清边界：视觉模型负责识别，GraphRAG 负责追因和知识增强
- 不要过度承诺多模态能力

---

## 24. 这个项目最大的难点是什么

### 问题

如果让你复盘，这个项目最大的技术难点是什么？

### 回答

最大的难点不是模型选型，而是怎么让系统**"既能回答，又尽量不胡说"**。

工业场景里用户最不接受的是看起来很专业、实际上没有依据的答案。所以真正难的是同时解决三件事：

1. **知识要组织得起来**：schema 设计、抽取质量、知识对齐
2. **证据要检索得回来**：子图挖掘、语义召回、上下文融合
3. **结果要可解释、可追溯**：证据链绑定、来源标注、置信度评估

我做的很多设计，例如保守抽取、条件化关系、关键实体抽取、子图模板化和稳定性优化，本质上都是围绕这个目标展开的。

### 追问

- 你最担心的错误类型是什么
- 如果知识图谱有噪声怎么办
- 怎么平衡召回和准确性
- 如果 LLM 生成胡说了怎么办

### 注意事项

- 这是一个很适合体现思考深度的问题，不要只答"数据难"或"模型难"
- 强调"可解释性"和"可追溯性"是工业场景的核心诉求

---

## 25. 这个方案的局限性是什么

### 问题

你觉得这个方案目前的局限性是什么？

### 回答

主要有三点。

**第一，图谱质量依赖上游数据**。如果文档脏、日志不规范、术语不统一，会影响抽取和对齐效果。数据清洗和术语归一需要持续投入。

**第二，因果关系来自经验知识**。很多因果链来自历史案例、专家经验和规则增强，它更接近工程上的高置信追因，不等于严格实验因果。对于新缺陷或未见过的参数组合，系统可能缺乏足够证据。

**第三，定位是辅助决策而非全自动**。一期更适合做辅助决策，不适合直接做无人审核的自动调参。我会把它定位成工艺工程师辅助系统，而不是全自动闭环控制系统。

### 追问

- 怎么进一步提升因果关系可信度
- 二期最想补的能力是什么
- 如果客户要求自动闭环怎么办
- 怎么扩展到新缺陷类型

### 注意事项

- 主动承认边界会显得更稳
- 不要把系统说成无所不能
- 可以提"二期方向"体现思考

---

## 26. 你的核心贡献是什么

### 问题

如果只总结你自己的核心贡献，你会怎么说？

### 回答

我自己的核心贡献可以概括成**三块**。

**第一，领域化改造**：围绕 TGV 场景完成 GraphRAG 的 schema 定制、抽取 prompt 改造和知识表示方式设计（三元组/四元组），构建 5 万级工艺-缺陷知识库。

**第二，推理引擎优化**：定位并缓解批量抽取场景下的推理服务瓶颈，将推理后端从 Ollama 迁移到 vLLM，通过投机解码等优化将吞吐提升 **38.6%**（300.6 → 416.6 tok/s）。

**第三，检索增强链路**：设计 Query NER 联合解码机制（< 1ms NER + Fuzzy Linking），实现 2-3 hop 子图挖掘与叙事化（上下文扩充 **+847%**），构建 HybridSearch 混合检索引擎，支撑带证据链的工艺追因和优化建议生成。

### 追问

- 哪块是你主导的，哪块是团队协作完成的
- 如果只保留一个亮点你会选哪个
- 你在这些工作中最大的成长是什么

### 注意事项

- 回答要聚焦你亲自做的事情
- 不要把团队成果全揽到自己身上
- 三个贡献点要清晰，便于面试官追问

---

## 27. 最后一句总结

### 问题

如果面试官让你最后再总结一下这个项目，你怎么讲？

### 回答

这个项目本质上是一个面向 TGV 制造场景的因果知识增强工艺评估系统。我的工作主要是三件事：第一，构建面向工业文档的数据清洗流程，定制 GraphRAG schema 和抽取策略，建立 5 万级工艺-缺陷知识库；第二，优化推理引擎稳定性，将 Ollama 迁移至 vLLM，通过投机解码将吞吐提升 38.6%；第三，设计 Query NER 联合解码、2-3 hop 子图挖掘和 HybridSearch 混合检索链路，实现上下文扩充 847%，支撑带证据链的缺陷追因和工艺优化建议生成。

### 追问

- 如果让你继续做二期，你最先做什么
- 这个项目能不能迁移到别的制造场景
- 你觉得这个项目的最大价值是什么

### 注意事项

- 最后一题不要展开太长，控制在 30 秒到 1 分钟
- 讲清"三件事"最有层次
- 三个数字要记熟：5 万级知识库、+38.6% 吞吐、+847% 上下文扩充
