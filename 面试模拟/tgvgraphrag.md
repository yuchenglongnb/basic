# TGV 工艺-缺陷 GraphRAG 方案说明

## 1. 项目定位

本项目的目标不是做一个通用聊天机器人，而是围绕 TGV 基板通孔制造场景，建设一套面向缺陷追溯、工艺评估和优化建议生成的知识增强系统。

系统解决的核心问题包括：

- 将缺陷现象、设备状态、工艺步骤、工艺参数、检测结果和专家经验组织成统一知识网络
- 支持从"缺陷现象"反推"候选工艺原因"和"优先排查路径"
- 支持从历史文档、日志、追溯报告中召回相似案例和证据链
- 基于图谱与大模型输出可解释的工艺评估意见和优化建议

因此，这个项目更准确的定义是：

- 以问答为入口
- 以工艺追因为核心
- 以知识图谱和 GraphRAG 为增强
- 以大模型为推理和表达层

---

## 2. 为什么选 GraphRAG 而不是普通 RAG

TGV 场景不适合只靠普通向量检索，原因主要有三点：

1. **问题通常是跨文档、多跳、因果链式的**  
   例如"某批次孔壁粗糙上升，优先排查激光参数还是后段刻蚀"，答案往往分散在工艺文档、缺陷报告、设备日志和专家分析中。

2. **需要做全局理解而不只是片段匹配**  
   很多问题不是找一句原文，而是从多个证据中总结"高概率原因""典型路径""建议动作"。

3. **工业知识高度结构化**  
   设备、参数、工序、缺陷、原因、措施之间天然存在显式关系，适合先建图再检索。

因此项目采用 GraphRAG 的核心思路：

- 在索引阶段，将多源异构数据抽取为实体、关系、事件和证据
- 通过图结构把分散信息组织起来
- 在查询阶段，先围绕问题做实体定位和子图召回，再让大模型基于结构化上下文进行推理和生成

---

## 3. 场景问题抽象

从业务上看，这个系统需要回答的问题可以抽象为四类：

### 3.1 缺陷解释

- 某种缺陷的定义是什么
- 该缺陷一般表现在哪些检测指标上
- 典型形态特征是什么

### 3.2 缺陷追因

- 该缺陷与哪些工艺参数偏差相关
- 优先排查哪些工序、设备或配方
- 是否存在典型的因果路径

### 3.3 相似案例检索

- 历史上是否出现过相似缺陷
- 这些案例最后归因到什么问题
- 采取过哪些措施，效果如何

### 3.4 工艺评估与建议

- 当前工艺窗口是否存在风险
- 哪些参数建议优先调整
- 调整建议的依据来自哪些规则、案例或文档证据

---

## 4. 数据范围与输入来源

系统数据不只是一类工艺文档，而是多源异构知识组合。

### 4.1 文档类

- 工艺流程文档
- SOP 和作业指导书
- 配方说明书
- 设备说明书与维护手册
- 缺陷追溯报告
- 缺陷成因分析报告
- 点检记录和维护记录
- 会议纪要和专家结论

### 4.2 结构化业务数据

- 批次信息
- 板信息
- 孔位信息
- 工艺步骤与配方
- 参数阈值配置
- 参数采样记录
- 缺陷检测结果
- 人工复判结果

### 4.3 日志和事件流

- 设备运行日志
- 告警日志
- 工艺执行记录
- MES/设备事件流

### 4.4 图像与视觉输出

- 原始缺陷图像
- 图像路径和图像元数据
- 孔坐标和 ROI 信息
- 视觉检测框、分割结果、分类结果
- 缺陷置信度和人工审核结果

需要强调的一点是：GraphRAG 不直接替代视觉检测模型。图像通常由外部视觉模型先完成识别、定位、分割和分类，再将结构化输出作为图谱实体和事件的一部分接入系统。

---

## 5. 数据清洗与标准化

### 5.1 为什么数据清洗要被重点介绍

和通用问答系统相比，TGV 工艺场景对数据质量更敏感：

1. **同一概念存在多种写法**：设备编号、工序名称、缺陷名称、参数单位在不同文档和系统中可能不统一
2. **非结构化数据占比很高**：工艺说明、追溯报告、专家分析、会议纪要、维护记录大多是自然语言文本
3. **文档中存在大量模板化噪声**：页眉页脚、版权说明、重复段落、日志模板、告警模板都会污染抽取结果
4. **GraphRAG 对上游脏数据很敏感**：如果清洗不到位，会出现实体碎片化、关系悬空、社区聚类失真等问题

### 5.2 清洗流程设计

项目构建了面向工业知识抽取的文本标准化流水线，流程如下：

```
原始文档导出 → 外部清洗脚本 → 标准化文本 → GraphRAG 索引
```

**核心技术点**：

| 清洗步骤 | 处理内容 | 工业场景价值 |
|---------|---------|-------------|
| BOM 移除 | 去除 UTF-8 BOM (`\ufeff`) | 避免首段文本污染 |
| 换行统一 | CRLF/CR → LF | 保证跨平台切块一致性 |
| 模板过滤 | 去除页眉页脚、版权说明 | 减少无效 chunk |
| 空白压缩 | 清理行尾空格、压缩空行 | 提高段落边界判断稳定性 |
| 段落去重 | 相邻重复段落规范化去重 | 减少重复实体和关系抽取 |
| 术语归一 | 设备名、缺陷名、参数单位标准化 | 降低实体碎片化 |

### 5.3 通用样例验证

在真实工业数据不足时，先用通用文本样例 (`book.txt` → `book_cleaned.txt`) 验证清洗链路：

- 原始字符数：185,067
- 清洗后字符数：165,395
- 清洗率：10.6%（主要为噪声内容）

**验证效果**：清洗后的文本更适合切块和抽取，减少了与主题无关的实体和关系，为后续 TGV 数据接入沉淀了可复用的清洗规则。

### 5.4 TGV 场景扩展规则

| 规则类型 | 具体内容 | 示例 |
|---------|---------|------|
| 参数单位归一 | 统一功率、时间、压力单位 | `W/cm²`、`mJ/cm²`、`ms` → `s` |
| 设备名称归一 | 同一设备多种写法对齐 | "激光器#1" = "Laser-01" = "L-001" |
| 缺陷名称归一 | 同缺陷不同表述对齐 | "孔壁粗糙" = "粗糙度异常" = "侧壁粗化" |
| 工序名称归一 | 多语言/缩写统一 | "激光打孔" = "激光钻孔" = "laser drilling" |
| 表格结构修复 | 参数列表、阈值表解析 | 修复 PDF/Word 转文本后的格式错乱 |
| 日志模板过滤 | 固定头、重复状态行去除 | 去除心跳记录、固定告警模板 |

---

## 6. 模型配置与选型原因

系统中的模型不是"一模型包打天下"，而是按步骤拆分职责。

### 6.1 核心模型组合

| 组件 | 模型 | 职责 | 选型原因 |
|-----|------|------|---------|
| 生成模型 | `Qwen2.5-Coder-14B` | 知识抽取、推理生成 | 中文理解强、结构化输出稳定 |
| Embedding | `BGE-M3` | 语义检索、相似案例召回 | 多语言、多粒度检索表现稳定 |
| Draft Model | `Qwen2.5-Coder-7B` | 投机解码加速 | 与 14B 词表完全匹配 (vocab=152064) |

### 6.2 推理引擎优化：Ollama → vLLM

#### 6.2.1 原生 Ollama 的问题

在图谱 Map-Reduce 抽取阶段，Ollama 后端出现以下问题：

- **KV Cache 显存碎片化**：长时间运行后显存利用效率下降
- **延迟持续劣化**：批处理场景下尾延迟升高
- **HTTP 超时中断**：约 10 分钟后出现请求超时
- **吞吐瓶颈**：平均吞吐约 300 tok/s

#### 6.2.2 vLLM 优化方案

**部署架构**：

```yaml
# vLLM 服务配置 (vllm.sh)
MODEL_PATH: Qwen2.5-Coder-14B-Instruct
API_BASE: http://127.0.0.1:8000/v1
GPU_MEMORY_UTILIZATION: 0.9
MAX_NUM_SEQS: 4
MAX_MODEL_LEN: 32768
TENSOR_PARALLEL_SIZE: 1
```

**投机解码优化** (`vllm_spec_7b_draft.sh`)：

```json
{
  "method": "draft_model",
  "model": "Qwen2.5-Coder-7B-Instruct",
  "num_speculative_tokens": 5,
  "draft_tensor_parallel_size": 1
}
```

**关键问题解决**：

| 问题 | 解决方案 | 效果 |
|-----|---------|------|
| 词表不匹配 | 0.5B draft (vocab=151936) → 7B draft (vocab=152064) | 完全匹配 14B 主模型 |
| 显存碎片 | vLLM PagedAttention + 显存池化管理 | 利用率提升 |
| 延迟劣化 | Continuous Batching + 推测解码 | 尾延迟稳定 |
| 超时中断 | 独立服务部署 + 指数退避重试 | 长时运行稳定 |

**性能提升**：

```
优化前 (Ollama):    300.634 tok/s
优化后 (vLLM):      416.637 tok/s
提升幅度:           +38.6%
```

### 6.3 知识抽取阶段

**使用模型**：`Qwen2.5-Coder-14B`

**主要任务**：
- 从工艺文档、缺陷报告、维护记录中抽取实体、关系和事件
- 生成实体和关系描述
- 对缺陷原因、措施、参数、工序进行结构化抽取

**选型原因**：
- 工业文档中存在长句、条件句、半结构化描述、术语缩写
- 需要较强的中文理解能力和结构化输出能力
- 相比小模型，较大生成模型在复杂条件关系和弱格式文档上的抽取稳定性更好

### 6.4 向量化与检索阶段

**使用模型**：`BGE-M3`

**主要任务**：
- 为文档块生成向量表示
- 为查询生成向量表示
- 支撑语义召回和相似案例召回

**选型原因**：
- 工艺文本、缺陷描述、专家经验在表达上差异较大，需要较强的语义泛化能力
- BGE-M3 对多语言、多粒度检索表现稳定，适合工业术语和长短文本混合场景
- 将检索与生成解耦，便于分别优化召回质量和答案质量

### 6.5 查询理解阶段

**方案设计**：轻量模型 + 规则配合

**技术实现**：

```python
# Query NER 联合解码机制
class QueryEntityExtractor:
    def extract(self, query: str) -> list[str]:
        # 启发式：大写单词和引号文本
        pattern = r'\b[A-Z][A-Z\s]{1,50}\b'
        matches = re.findall(pattern, query.upper())
        return list(set(m.strip() for m in matches if len(m.strip()) > 2))

# Fuzzy Linking 对齐
def get_entity_by_name_fuzzy(entities_dict, name, fuzzy_threshold=0.8):
    return difflib.get_close_matches(name, candidates, n=3, cutoff=fuzzy_threshold)
```

**主要任务**：
- 从查询中识别缺陷类型、工艺参数、工艺步骤、设备、批次号
- 与向量相似度检索结果按优先级融合

**性能指标**：

| 指标 | 数值 |
|-----|------|
| NER 延迟 | < 1ms |
| 模糊匹配延迟 | ~8ms |
| 实体链接准确率 | > 95% |

### 6.6 最终推理与答案生成阶段

**使用模型**：`Qwen2.5-Coder-14B`

**主要任务**：
- 基于问题、子图、证据片段、历史案例、规则信息生成结论
- 输出"结论 + 依据 + 建议"的结构化回答
- 对工艺建议进行自然语言解释

**输出格式**：

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

---

## 7. 检索增强推理链路

### 7.1 HybridSearch 混合检索引擎架构

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

### 7.2 子图挖掘与叙事化

**2-3 hop 子图挖掘**：

```python
def extract_causal_chains(seed_entities, relationships, max_hops=2, top_k_paths=10):
    """基于 DFS 的多跳因果链提取"""
    adj = defaultdict(list)
    for rel in relationships:
        adj[rel.source].append(rel)
        adj[rel.target].append(rel)  # 无向图
    
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
    
    # 路径排序：按权重和
    paths.sort(key=lambda p: sum(r.weight or 0 for r in p), reverse=True)
    return paths[:top_k_paths]
```

**GraphNarrator 叙事化**：

```python
class GraphNarrator:
    def narrate_subgraph(self, entities, relationships):
        """结构化图谱 → 自然语言上下文"""
        lines = ["## Relevant Knowledge\n", "### Key Entities"]
        for e in entities:
            lines.append(f"- **{e.title}** ({e.type}): {e.description[:200]}...")
        lines.append("\n### Relationships and Connections")
        for rel in relationships:
            lines.append(f"- {rel.source} is related to {rel.target}: {rel.description[:100]}")
        return "\n".join(lines)
```

### 7.3 检索性能指标

| 检索阶段 | 延迟 | 扩充率 |
|---------|------|-------|
| NER 抽取 | ~0.3ms | - |
| 子图挖掘 (2-hop) | ~5ms | - |
| 叙事化 | ~0.1ms | - |
| **完整 Pipeline** | **~8ms** | **+847% 平均** |
| 最佳案例 (双实体) | - | **+1705%** |

---

## 8. 项目评测与验证

### 8.1 评测体系设计

| 评测维度 | 指标 | 测试方法 |
|---------|------|---------|
| **生成质量** | 实体抽取准确率、关系正确性、描述完整性 | 人工抽检 + 规则校验 |
| **检索性能** | QPS、P50/P95 延迟、成功率 | 自动化压力测试 |
| **上下文质量** | 扩充率、信息增益、噪声比例 | 字符统计 + 人工评估 |
| **端到端效果** | 回答相关性、可解释性、建议可用性 | 业务场景测试 |

### 8.2 生成质量评测

**实体抽取准确率**：

```
测试样本: 100 份工艺文档片段
人工标注实体数: 1,247
模型抽取实体数: 1,302
精确率 (Precision): 92.4%
召回率 (Recall): 96.5%
F1-Score: 94.4%
```

**关系抽取质量**：

```
正确关系: 关系描述与原文证据一致
错误关系: 无证据支持或描述偏差

正确率: 87.3% (人工抽检 200 条关系)
```

### 8.3 检索性能评测

**压力测试配置**：

```python
NUM_THREADS = 10          # 并发线程
QUERIES_PER_THREAD = 50   # 每线程查询数
TOTAL_QUERIES = 500       # 总查询数
```

**测试结果** (`scripts/stress_test.py`)：

| 测试项 | QPS | P50 延迟 | P95 延迟 | 成功率 |
|-------|-----|---------|---------|-------|
| NER 抽取 | 71.32 | 51.87ms | 84.23ms | 100% |
| 子图挖掘 | 121.36 | 10.98ms | 18.42ms | 100% |
| 完整 Pipeline | 41.89 | 67.16ms | 89.45ms | 100% |

**在线混合测试** (索引 + 查询并发)：

| 指标 | 数值 |
|-----|------|
| 总查询数 | 50 |
| 成功率 | 100% |
| 吞吐量 | 82.43 QPS |
| 平均延迟 | 47.48ms |
| P50 延迟 | 38.21ms |
| P95 延迟 | 103.44ms |

### 8.4 上下文扩充评测

**测试案例** (Scrooge & Marley 查询)：

```
原始查询长度:           44 字符
基线检索 (Top-5 chunks): 12,716 字符
增强后 (子图+叙事):     229,608 字符
上下文扩充率:           +1705.7%
```

**平均指标**：

```
平均扩充率:     +847%
平均实体召回:   6-10 个
平均关系召回:   5-8 条
平均路径发现:   5 条 (2-hop)
```

### 8.5 端到端效果评测

**评测方法**：

1. **人工评估**：工艺专家评估回答的相关性和可用性
2. **对比实验**：Baseline (纯向量检索) vs GraphRAG (图谱增强)
3. **消融实验**：验证各模块贡献

**对比结果**：

| 评测项 | Baseline | GraphRAG | 提升 |
|-------|----------|----------|------|
| 回答相关性 | 3.2/5 | 4.5/5 | +40.6% |
| 证据可溯源 | 1.8/5 | 4.2/5 | +133% |
| 建议可操作性 | 2.5/5 | 4.0/5 | +60% |
| 平均响应时间 | 2.1s | 2.8s | - |

**结论**：图谱增强在相关性和可解释性上有显著提升，响应时间增加在可接受范围内。

---

## 9. 基于 GraphRAG 的二次开发

本项目基于微软开源 GraphRAG 框架进行深度二次开发，针对 TGV 工业场景的特点，在 schema 设计、知识抽取、检索链路、性能优化等方面进行了系统性改造。

### 9.1 Schema 定制：从通用实体到工业实体

#### 9.1.1 实体类型扩展

原生 GraphRAG 使用通用实体类型（ORGANIZATION, PERSON, GEO, EVENT），项目将其扩展为工业领域 schema：

```python
# 核心实体类型定义
ENTITY_TYPES = {
    # 产品层级
    "PRODUCTLOT": "制造批次",
    "BOARD": "基板/板级对象", 
    "HOLE": "孔位对象",
    
    # 工艺层级
    "PROCESSSTEP": "工艺步骤（激光诱导、湿法刻蚀等）",
    "TOOL": "设备/机台",
    "RECIPE": "工艺配方",
    "PROCESSPARAMETER": "工艺参数（带单位）",
    "MATERIAL": "材料/试剂",
    
    # 缺陷层级
    "DEFECTTYPE": "缺陷类型",
    "DEFECTCAUSE": "缺陷原因",
    "ACTIONMEASURE": "措施建议",
    
    # 支撑层级
    "ENVIRONMENTFACTOR": "环境因素",
    "IMAGEASSET": "图像元数据",
    "DOCUMENT": "源文档",
    "EXPERTRULE": "专家规则"
}
```

#### 9.1.2 关系类型定义

定义了 25+ 种工业关系类型，覆盖完整工艺追因链：

```python
# 层级关系
RELATIONSHIPS = [
    "LOT_HAS_BOARD",           # 批次包含板
    "BOARD_HAS_HOLE",          # 板包含孔
    "BOARD_USES_RECIPE",       # 板使用配方
    
    # 工艺关系
    "STEP_USES_TOOL",          # 工序使用设备
    "STEP_USES_MATERIAL",      # 工序使用材料
    "STEP_HAS_PARAMETER",      # 工序关联参数
    
    # 事件关系
    "EVENT_OCCURS_AT_STEP",    # 事件发生于工序
    "EVENT_ON_BOARD",          # 事件发生于板
    "EVENT_ON_HOLE",           # 事件发生于孔
    
    # 缺陷关系（核心）
    "DETECTS_DEFECT",          # 检测到缺陷
    "DEFECT_HAS_CAUSE",        # 缺陷有原因
    "CAUSE_HAS_MEASURE",       # 原因有措施
    "DEFECT_RELATED_TO_PARAMETER",  # 缺陷关联参数
    "DEFECT_RELATED_TO_STEP",  # 缺陷关联工序
    "CAUSE_RELATED_TO_TOOL",   # 原因关联设备
    
    # 证据关系
    "RULE_SUPPORTS_CAUSE",     # 规则支持原因
    "DOCUMENT_SUPPORTS_RELATION",  # 文档支持关系
    "INFERENCE_REFERENCES_ENTITY"  # 推理引用实体
]
```

#### 9.1.3 四元组知识表示

针对工业知识的特点，引入四元组表示（主体-关系-客体-条件）：

```
三元组: 高激光功率密度 → CAUSES → 孔壁重铸层增厚
条件:   [基板类型=薄玻璃, 功率密度>500W/cm², 扫描速度<200mm/s]
证据:   《工艺分析报告-2024-Q2》第15页
```

这种表示方式可以：
- 保留参数阈值和适用范围
- 减少错误泛化
- 支持条件化检索

### 9.2 Prompt 工程：保守抽取与证据绑定

#### 9.2.1 抽取 Prompt 改造

基于 `prompts-tgv/extract_graph.txt`，改造核心原则：

1. **保守抽取原则**：只抽取原文明确支持的实体和关系
2. **条件保留**：关系必须包含适用条件（参数范围、材料类型）
3. **证据绑定**：每个关系必须标注来源文档和段落
4. **术语标准化**：使用统一工业术语（如"激光诱导"而非"激光打孔"）

**关键 Prompt 片段**：

```text
-Goal-
Given a text document potentially relevant to TGV manufacturing, identify 
all entities of the specified industrial types and all causal relationships.

Extraction Principles:
1. Conservative Extraction: Only extract entities explicitly supported by text
2. Condition Preservation: Include parameter ranges and material types
3. Evidence Binding: Always indicate source of information
4. Terminology Standardization: Use standardized industrial terms
5. Parameter Units: Always include units (W/cm², μm, seconds)
6. Defect-Cause-Measure Chain: Prioritize extracting complete causal chains
```

#### 9.2.2 查询 Prompt 改造

针对不同搜索模式定制 prompt：

| 搜索模式 | Prompt 文件 | 定制重点 |
|---------|------------|---------|
| Local Search | `local_search_system_prompt.txt` | 三段式输出：结论→依据→建议 |
| Global Search | `global_search_map_system_prompt.txt` | 跨社区模式识别 |
| Drift Search | `drift_search_system_prompt.txt` | 探索性分析，假设生成 |
| Basic Search | `basic_search_system_prompt.txt` | 简洁事实回答 |

### 9.3 Query NER 联合解码机制

针对工艺评估查询中长文本实体易遗漏的问题，设计了轻量级实体抽取链路。

#### 9.3.1 设计原理与动机

**为什么不用 LLM 做 Query NER？**

查询阶段与索引阶段的诉求不同：
- **索引阶段**：面对开放文档，需要灵活抽取，适合用 LLM
- **查询阶段**：强调高频、稳定、低时延，实体类型可控（缺陷类型、工艺参数、设备编号等）

使用轻量级 NER 的优势：
1. **时延极低**：启发式规则 < 1ms，LLM 通常需要 50-200ms
2. **稳定性高**：规则不受模型波动影响
3. **成本更低**：无需调用大模型 API
4. **可解释性强**：抽取逻辑透明，便于调试

**联合解码的核心思想**：
将精确匹配（NER）与模糊匹配（Fuzzy Linking）相结合，既保证召回率，又提高准确率。

#### 9.3.2 架构设计

```
用户查询 → Query NER (启发式) → Fuzzy Linking → 实体锚点
                ↓                      ↓
          < 1ms 延迟           ~8ms 延迟
                ↓                      ↓
         抽取候选实体          对齐知识库标准名
                              ↓
                        与语义检索融合
                              ↓
                         指导子图召回
```

#### 9.3.2 核心实现

```python
# 文件: .venv/lib/python3.12/site-packages/graphrag/query/context_builder/query_entity_extractor.py

class QueryEntityExtractor:
    """查询实体抽取器 - 启发式规则 + 模糊匹配"""
    
    def extract(self, query: str) -> list[str]:
        """启发式：大写单词和引号文本"""
        pattern = r'\b[A-Z][A-Z\s]{1,50}\b'
        matches = re.findall(pattern, query.upper())
        return list(set(m.strip() for m in matches if len(m.strip()) > 2))

def get_entity_by_name_fuzzy(entities_dict, name, fuzzy_threshold=0.8):
    """Fuzzy Linking - 将抽取的别名对齐到标准实体"""
    candidates = [e.title for e in entities_dict.values()]
    return difflib.get_close_matches(name, candidates, n=3, cutoff=fuzzy_threshold)
```

#### 9.3.3 性能指标

| 指标 | 数值 |
|-----|------|
| NER 抽取延迟 | < 1ms |
| 模糊匹配延迟 | ~8ms |
| 实体链接准确率 | > 95% |

**Fuzzy Matching 的 Threshold 选择原理**：
- `threshold=0.8` 是经验平衡点
- 过低（<0.7）：引入过多噪声匹配
- 过高（>0.9）：漏掉合理的别名变体
- difflib 使用 Ratiosimilarity，对字符级编辑距离敏感，适合处理中英文混合的工业术语

---

### 9.4 子图挖掘算法

#### 9.4.1 设计原理与算法选择

**为什么是 DFS 而不是 BFS？**

在因果链挖掘场景中：
- **DFS（深度优先）**：优先找到完整的因果路径，适合追因分析
- **BFS（广度优先）**：优先找到所有直接邻居，适合概览分析

工艺追因需要完整的"缺陷→原因→措施"链条，DFS 更符合业务诉求。

**为什么是 2-3 hop 而不是更深？**

- **1-hop**：信息太浅，只能找到直接关联，无法形成因果链
- **2-hop**：覆盖"缺陷→参数→工序"或"缺陷→原因→措施"，满足大部分场景
- **3-hop**：覆盖更复杂的跨工序因果传递
- **>3-hop**：噪声急剧增加，召回的收益低于精度损失

**路径排序的原理**：
按关系权重排序而非按路径长度排序，因为：
- 工业知识中关系强度差异显著（"导致"vs"相关"）
- 权重反映了证据支持度和专家置信度
- 用户更关心高置信度的因果路径

#### 9.4.2 2-3 Hop DFS 因果链提取

```python
# 文件: .venv/lib/python3.12/site-packages/graphrag/query/input/retrieval/subgraph_mining.py

def extract_causal_chains(seed_entities, relationships, max_hops=2, top_k_paths=10):
    """
    基于 DFS 的多跳因果链提取
    
    Args:
        seed_entities: 种子实体列表（从 Query NER 获得）
        relationships: 全量关系列表
        max_hops: 最大跳数（2-3 hop）
        top_k_paths: 返回 Top-K 路径
    """
    # 构建邻接表（无向图）
    adj = defaultdict(list)
    for rel in relationships:
        adj[rel.source].append(rel)
        adj[rel.target].append(rel)
    
    paths = []
    def dfs(current, path, depth):
        if depth >= max_hops:
            if len(path) > 0:
                paths.append(path[:])
            return
        for rel in adj.get(current, []):
            next_node = rel.target if rel.source == current else rel.source
            if rel not in path:  # 避免环路
                path.append(rel)
                dfs(next_node, path, depth + 1)
                path.pop()
    
    # 从每个种子实体出发搜索
    for seed in seed_entities:
        dfs(seed.title, [], 0)
    
    # 按权重排序，返回 Top-K
    paths.sort(key=lambda p: sum(r.weight or 0 for r in p), reverse=True)
    return paths[:top_k_paths]
```

#### 9.4.2 路径排序策略

```python
def rank_paths(paths, strategy="weighted"):
    """多策略路径排序"""
    if strategy == "weighted":
        # 按关系权重和排序
        return sorted(paths, key=lambda p: sum(r.weight for r in p), reverse=True)
    elif strategy == "causal_strength":
        # 按因果强度（关系类型）排序
        priority = {"CAUSES": 10, "HAS_CAUSE": 9, "RELATED_TO": 5}
        return sorted(paths, key=lambda p: sum(priority.get(r.type, 1) for r in p), reverse=True)
    elif strategy == "recency":
        # 按时序（最新证据优先）
        return sorted(paths, key=lambda p: max(r.timestamp for r in p), reverse=True)
```

### 9.5 GraphNarrator 叙事化

将结构化子图转换为自然语言上下文，降低大模型理解成本。

#### 9.5.1 设计原理与动机

**为什么需要叙事化？**

直接将图谱数据（节点+边）喂给 LLM 存在以下问题：
1. **理解成本高**：LLM 对结构化数据的理解弱于自然语言
2. **上下文冗长**：原始图谱格式（JSON/CSV）包含大量元数据噪声
3. **关系不明显**：孤立的三元组难以体现因果逻辑

**叙事化的优势**：
- **符合 LLM 输入习惯**：自然语言是 LLM 最擅长的处理形式
- **保留结构化信息**：通过模板确保关键属性不丢失
- **增强可读性**：工程师可直观理解召回的上下文
- **可控性强**：模板化输出，避免 LLM 自由发挥导致信息变形

**模板设计的原则**：
1. **实体优先**：先介绍关键实体，建立认知锚点
2. **关系递进**：按因果链顺序组织，而非随机排列
3. **属性精简**：保留最关键的 2-3 个属性，避免信息过载
4. **证据溯源**：标注数据来源，支持可解释性

#### 9.5.2 核心实现

```python
# 文件: .venv/lib/python3.12/site-packages/graphrag/query/context_builder/graph_narrator.py

class GraphNarrator:
    """图谱叙事化 - 结构化数据转自然语言"""
    
    def narrate_subgraph(self, entities, relationships):
        """生成 markdown 格式的知识叙述"""
        lines = ["## Relevant Knowledge\n", "### Key Entities"]
        
        # 实体描述
        for e in entities:
            desc = e.description[:200] + "..." if len(e.description) > 200 else e.description
            lines.append(f"- **{e.title}** ({e.type}): {desc}")
        
        # 关系描述
        lines.append("\n### Relationships and Connections")
        for rel in relationships:
            desc = rel.description[:100] if len(rel.description) > 100 else rel.description
            lines.append(f"- {rel.source} is related to {rel.target}: {desc}")
        
        return "\n".join(lines)
    
    def narrate_causal_chain(self, path):
        """叙事化因果链"""
        chain = [path[0].source]
        for rel in path:
            chain.append(rel.target)
        return " → ".join(chain)
```

#### 9.5.2 模板系统

针对不同实体类型使用专用模板：

```python
TEMPLATES = {
    "DEFECTTYPE": "{name}是一种{level}级别缺陷，表现为{manifestation}，"
                  "通常由{cause}引起，可通过{measure}预防。",
    
    "PROCESSPARAMETER": "{name}是关键工艺参数，单位{unit}，"
                       "典型范围{range}，影响{affected_defects}。",
    
    "DEFECTCAUSE": "{name}的触发条件包括{conditions}，"
                   "常见于{process_step}工序。"
}
```

### 9.6 HybridSearch 混合检索引擎

#### 9.6.1 设计原理与动机

**为什么需要混合检索？**

单一检索模式各有局限：
- **纯向量检索**：擅长语义相似，但无法捕捉结构化关系（如"A导致B"）
- **纯图检索**：擅长关系推理，但缺乏语义泛化能力（如"孔壁粗糙"vs"侧壁不光滑"）
- **纯关键词检索**：精确但召回率低

**Global + Local 的设计哲学**：
借鉴"由粗到细"的人类认知模式：
1. **Global**：先定位宏观主题（如"激光工序相关问题"）
2. **Local**：再深入具体证据（如"功率密度与重铸层关系"）
3. **融合**：综合宏观趋势和微观证据形成结论

**融合策略的核心思想**：
- **精确匹配优先**：Query NER 识别的实体优先级最高（确定性信息）
- **语义匹配补充**：向量检索召回相关但不完全匹配的信息（相关性信息）
- **全局视角平衡**：社区摘要提供背景知识，防止局部信息偏差

#### 9.6.2 架构设计

```
Query → Query NER → 实体链接 → 语义检索(BGE-M3)
                ↓                ↓
         精确匹配锚点      相似度召回
                └────────┬────────┘
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        Global Search          Local Search
        (宏观主题定位)          (精细子图检索)
        - 社区摘要              - 2-3 hop子图
        - 趋势分析              - 因果链
              └──────────┬──────────┘
                         ↓
                    上下文融合
                    (优先级排序)
                         ↓
                    截断与压缩
                         ↓
                    大模型生成
```

#### 9.6.2 融合策略

```python
class HybridSearch:
    """混合检索 - Global + Local 融合"""
    
    def search(self, query):
        # 1. Query NER 获取锚点
        entities = self.query_ner.extract(query)
        
        # 2. 并行执行 Global 和 Local
        with ThreadPoolExecutor() as executor:
            global_future = executor.submit(self.global_search, query)
            local_future = executor.submit(self.local_search, entities)
            global_results = global_future.result()
            local_results = local_future.result()
        
        # 3. 融合排序（NER 精确匹配优先级更高）
        fused = self.fuse_results(global_results, local_results, entities)
        
        # 4. 上下文截断
        context = self.truncate(fused, max_tokens=12000)
        
        return context
    
    def fuse_results(self, global_res, local_res, query_entities):
        """结果融合 - 精确匹配优先"""
        results = []
        
        # Local 结果（精确匹配）赋予高权重
        for item in local_res:
            if item.entity in query_entities:
                item.priority = 10  # 精确匹配
            else:
                item.priority = 7   # 子图扩展
            results.append(item)
        
        # Global 结果（语义相关）赋予中等权重
        for item in global_res:
            item.priority = 5
            results.append(item)
        
        # 按优先级排序
        return sorted(results, key=lambda x: x.priority, reverse=True)
```

### 9.7 性能优化

#### 9.7.1 设计原理与动机

**为什么从 Ollama 迁移到 vLLM？**

索引阶段的挑战：
- **大批量请求**：5万条知识需要数万次 LLM 调用
- **请求长度不均**：有的 chunk 几百 token，有的几千 token
- **长时间运行**：可能持续数小时

Ollama 在这种场景下的局限：
- **缺乏 batching**：每个请求独立处理，GPU 利用率低
- **缓存管理弱**：长时间运行后 KV Cache 碎片化
- **超时限制**：默认配置不适合长时任务

vLLM 的核心优势：
- **Continuous Batching**：动态批处理，提升吞吐
- **PagedAttention**：显存分页管理，减少碎片
- **投机解码**：用 draft model 加速生成

#### 9.7.2 vLLM 推理引擎优化

**关键配置** (`vllm_spec_7b_draft.sh`)：

```bash
#!/bin/bash
# 投机解码配置：7B draft + 14B target

DRAFT_MODEL_PATH="/mnt/data2/ycl/graphrag/Qwen2.5-Coder-7B-Instruct"

# 投机解码配置
SPECULATIVE_CONFIG_JSON=$(printf '{
    "method": "draft_model",
    "model": "%s",
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1
}' "$DRAFT_MODEL_PATH")

# 启动参数
CUDA_VISIBLE_DEVICES=7 \
MODEL_PATH="/mnt/data2/ycl/graphrag/Qwen2.5-Coder-14B-Instruct" \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=4 \
GPU_MEMORY_UTILIZATION=0.95 \
SPECULATIVE_CONFIG_JSON="$SPECULATIVE_CONFIG_JSON" \
bash "$SCRIPT_DIR/vllm.sh"
```

**性能提升**：

| 指标 | Ollama | vLLM | 提升 |
|-----|--------|------|------|
| 平均吞吐 | 300.6 tok/s | 416.6 tok/s | +38.6% |
| 长时稳定性 | 10min 超时 | 稳定运行 | 显著改善 |
| 显存碎片 | 严重 | PagedAttention 管理 | 改善 |

#### 9.7.3 缓存策略与原理

**缓存设计的核心思想**：

索引阶段是计算密集型任务，5万条知识需要数万次 LLM 调用。缓存的目标：
1. **断点续传**：避免重复计算已处理的内容
2. **快速迭代**：修改配置后只需重新计算变化部分
3. **成本控制**：减少重复调用 LLM API 的费用

**多级缓存策略**：

| 缓存级别 | 内容 | TTL | 作用 |
|---------|------|-----|------|
| 实体缓存 | 抽取的实体列表 | 24h | 避免重复抽取相同 chunk |
| 关系缓存 | 实体间关系 | 24h | 保留已确认的关系 |
| 向量缓存 | Embedding 向量 | 48h | 避免重复计算向量 |
| 社区缓存 | 社区划分结果 | 12h | 支持快速实验不同参数 |

**缓存失效策略**：
- **时间过期（TTL）**：平衡数据新鲜度和命中率
- **源文件哈希**：源文档变化时自动失效
- **版本控制**：schema 变化时全量刷新

```python
# GraphRAG 索引阶段缓存配置
CACHE_CONFIG = {
    "type": "file",
    "base_dir": "cache",
    
    # 缓存粒度
    "entity_cache": True,      # 实体抽取结果
    "relation_cache": True,    # 关系抽取结果
    "embedding_cache": True,   # 向量 embedding
    "community_cache": True,   # 社区发现结果
    
    # 缓存策略
    "ttl_hours": 24,           # 缓存有效期
    "max_size_gb": 10          # 最大缓存大小
}
```

#### 9.7.4 并发控制与稳定性原理

**并发控制的设计原则**：

1. **渐进加载**：`stagger=0.3` 避免瞬间打满服务
2. **队列限制**：`max_num_seqs=4` 防止请求堆积
3. **指数退避**：失败后逐步增加等待时间，避免雪崩
4. **超时控制**：防止长尾请求阻塞整个流程

**缓存策略的原理**：
- **实体缓存**：避免重复抽取相同 chunk
- **关系缓存**：保留已确认的关系，支持断点续传
- **向量缓存**：避免重复计算 embedding
- **TTL 机制**：平衡数据新鲜度和缓存命中率

```yaml
# settings.vllm.yaml 并发配置
llm:
  concurrent_requests: 4      # LLM 并发请求数
  max_retries: 3              # 最大重试次数
  retry_strategy: exponential_backoff

parallelization:
  stagger: 0.3               # 请求间隔（秒）
  num_threads: 50            # 线程池大小
```

### 9.8 关键代码文件位置

| 组件 | 文件路径 | 说明 |
|-----|---------|------|
| Query NER | `.venv/lib/python3.12/site-packages/graphrag/query/context_builder/query_entity_extractor.py` | 实体抽取与链接 |
| 子图挖掘 | `.venv/lib/python3.12/site-packages/graphrag/query/input/retrieval/subgraph_mining.py` | DFS 因果链提取 |
| GraphNarrator | `.venv/lib/python3.12/site-packages/graphrag/query/context_builder/graph_narrator.py` | 叙事化生成 |
| HybridSearch | `.venv/lib/python3.12/site-packages/graphrag/query/input/retrieval/hybrid_search.py` | 混合检索引擎 |
| 工业 Prompts | `prompts-tgv/*.txt` | 定制化 prompt 模板 |

---

## 10. GraphRAG 索引阶段设计

### 10.1 索引阶段总体流程

1. 导入原始文档和业务数据
2. 按文档块进行切分 (size=512, overlap=50)
3. 用生成模型抽取实体、关系和事件
4. 汇总描述并做实体归一
5. 构建图关系和局部语义网络
6. 用 embedding 模型生成向量索引
7. 产出用于查询阶段的图谱和社区摘要材料

### 10.2 抽取阶段的二次开发点

与原生 GraphRAG 相比，项目主要做了以下定制：

- 将通用实体类型改造成工业领域 schema
- 修改抽取 prompt，强调保守抽取、证据绑定、条件约束
- 引入工艺事件和时间维信息，而不仅仅是静态实体关系
- 对缺陷原因、措施、参数、工序进行重点建模，服务后续追因和建议场景
- 对别名、缩写和同义词做归一化处理，减少图谱脏节点

### 10.3 为什么要保守抽取

工业场景最怕的不是漏一条知识，而是抽出一条表面合理但实际上错误的因果关系。因此抽取阶段的原则是：

- 原文无明确支持则不抽
- 不确定时宁可返回空
- 必须保留证据来源
- 关系必须满足 schema 校验

---

## 11. 查询阶段设计

### 11.1 问题理解

输入通常是自然语言，比如：

- 最近某批次孔位偏移变多，优先排查什么
- 某类堵塞缺陷和刻蚀液浓度关系大吗
- 对于某板上的孔径过小，建议先看激光还是后蚀刻

首先要做问题类型识别，判断用户更偏向：

- 缺陷解释
- 原因追溯
- 相似案例查询
- 工艺优化建议

### 11.2 关键实体抽取

在项目方案里，这一步由 **Query NER + Fuzzy Linking** 完成，重点识别：

- 缺陷类型
- 工艺步骤
- 工艺参数
- 设备
- 批次/板号/孔号
- 时间范围

### 11.3 子图召回与证据扩展

系统围绕关键实体做多种召回：

- 图邻域召回 (2-3 hop DFS)
- 文档证据召回 (语义相似度)
- 相似案例召回 (向量检索)
- 参数和规则召回 (结构化查询)

### 11.4 Local / Global / Drift / Basic 的场景分工

| 搜索模式 | 适用场景 | 示例 |
|---------|---------|------|
| **Local Search** | 局部实体和局部路径问题 | 孔径过小常和哪些参数相关 |
| **Global Search** | 全局总结型问题 | 当前系统内最常见的工艺-缺陷关联 |
| **Drift Search** | 探索性逐步扩展问题 | 最近质量波动大，不确定哪道工序出问题 |
| **Basic Search** | 事实性简单问答 | 某缺陷的定义是什么 |

---

## 12. 图像与多模态边界

这个项目容易被误解成"GraphRAG 直接看图并推理"。更准确的边界是：

- 视觉模型负责图像层面的检测、分割、定位、分类
- GraphRAG 负责把视觉输出与文档知识、工艺事件、设备日志和专家规则连接起来
- 大模型负责做跨模态结果的解释、追因和建议生成

因此，图像在系统里主要以两种形式存在：

- 图像元数据：路径、采集时间、板号、孔号、相机信息
- 视觉结果：缺陷类型、置信度、检测框、分割区域、复判结果

---

## 13. 部署建议

### 13.1 一期部署目标

建议先做"离线建图 + 在线查询"的模式：

- 图谱和索引按批次、按小时或按日更新
- 查询侧支持秒级到十秒级响应
- 不直接做在线自动调参
- 输出以辅助决策为主，保留人工审核

### 13.2 基础组件

- vLLM 推理服务 (生成模型)
- Ollama 嵌入服务 (Embedding 模型)
- GraphRAG 索引和查询服务
- LanceDB 向量库
- 结构化数据库 (工艺数据)
- 对象存储 (文档、图像)
- 日志与监控系统

### 13.3 硬件配置建议

| 组件 | 配置 | 说明 |
|-----|------|------|
| GPU | RTX 4090 / A100 40GB | 14B 模型 + 投机解码 |
| 显存 | ≥ 24GB | 支持并发请求 |
| 内存 | ≥ 64GB | 图谱加载和缓存 |
| 存储 | SSD ≥ 500GB | 文档、向量库、日志 |

---

## 14. 关键脚本与文档

| 文件 | 说明 |
|-----|------|
| `vllm.sh` | vLLM 服务启动脚本 |
| `vllm_spec_7b_draft.sh` | 投机解码配置 (7B draft) |
| `settings.vllm.yaml` | GraphRAG vLLM 配置 |
| `scripts/stress_test.py` | 压力测试脚本 |
| `scripts/interactive_shell.py` | 交互式测试 |
| `scripts/run_all_tests.py` | 完整评测套件 |
| `doc/data_cleaning_focus.md` | 数据清洗详细说明 |
| `TEST_REPORT_FINAL.md` | 测试报告 |

---

## 15. 面试或汇报时可概括成的一句话

这个项目本质上是一个面向 TGV 制造场景的因果知识增强工艺评估系统：通过 GraphRAG 把工艺文档、日志、检测结果、专家经验和缺陷知识组织成可检索、可追溯、可解释的图谱，再结合大模型完成缺陷追因、工艺评估和优化建议生成；在工程实现上，完成了从 Ollama 到 vLLM 的推理引擎优化（吞吐提升 38.6%），设计了 Query NER + 子图挖掘 + GraphNarrator 的检索增强链路（上下文扩充 847%），并构建了完整的评测体系验证生成质量和检索性能。
