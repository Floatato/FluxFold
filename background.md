# Background

本文档梳理现有 Agent memory 系统在**记忆组织**、**关联发现**、**检索**和**演化**四个方面的共同局限，并说明 FluxFold 的设计如何回应这些局限。所有对参考工作的判断都以其论文报告的数值、消融设置或仓库中的实际实现为依据；无法从公开材料证实的推断会明确标注为推断。

一个前置说明：Agent memory 包含两个目标并不相同的过程。**记忆的添加与整理**关心如何从持续到来的交互中抽取信息、消解冲突并建立联系；**记忆的检索**关心如何针对当前 query，在有限的时间和上下文预算内找到足以支持回答的证据。一个便于整理和解释的存储结构，并不必然对应一个高效、准确的检索结构。下文的多数张力都源于这两个过程被同一套数据结构同时承担。

---

## 1. 组织结构：固定分类与动态聚类

### 1.1 固定记忆类型的实际作用边界

一类工作按认知科学的记忆分类给记忆划分类型：MIRIX 分为 Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault 六类，各配一个 Memory Manager，由 Meta Memory Manager 路由；Hindsight 分为 world、experience、opinion、observation 四个 network；Nemori 分为 Episodic 与 Semantic 两层；MemOS 用 `memory_type` 区分 WorkingMemory / LongTermMemory / UserMemory，并另设 PreferenceTextMemory。

这些划分在**写入时**确实提供了不同的抽取模板：Episodic 保留时间戳与事件细节，Procedural 保留步骤序列，Knowledge Vault 对敏感值做访问控制。问题出现在划分之后：

**类型是标签，不是有界容器。** 记忆数量随交互无限增长，而类型数量固定，因此“某一类记忆”会持续膨胀成一个没有上界的集合。这带来一个直接后果：**系统失去了可以整体审计的粒度单元。** 想要发现并消解重复、冲突和状态变化，只能退回到“以新记忆为 query 做相似度召回，再在召回结果内部整理”，而这恰恰把整理能力限制在了相似度的能力边界内（见第 2 节）。MIRIX、Hindsight、MemOS 都没有定义“对某个记忆子集做一次完整复核”的操作，因为不存在规模有界的子集。

**类型划分的可靠性依赖抽取阶段的判断。** Nemori 的 `memory_type`（identity / preference / relationship / goal / belief / habit）在当前实现中由关键词启发式决定，`confidence` 生成时恒为 `1.0`；这两个字段既不参与检索，也不参与整理。Hindsight 论文的四网络中，`opinion` 网络与 `confidence_score` 是其对比表（Table 1）中相对于 Zep、A-Mem、Mem0 的主要差异项，但最新代码已经移除了 `opinion` 与 `confidence_score`，主观知识改由 observation 巩固和用户策展的 `mental_models` 承担。也就是说，**论文用以论证分类必要性的那一层，在实现中并未保留。**

### 1.2 分类收益的实验证据普遍与检索预算混淆

多类记忆同时召回的得分高于单类召回，是这些工作论证分类价值的主要证据。但这类比较通常没有固定检索预算，因此无法区分“分类本身有价值”和“召回了更多条目”。

以 Nemori 为例，其主实验固定 `k = 10` 个 episode 加 `m = 2k = 20` 条 semantic，共 30 个条目。消融时：

- `w/o e`（去掉 episodic 检索）实际是只召回 20 条 semantic，得分 0.744 → 0.615；
- `w/o s`（去掉 semantic 检索）实际是只召回 10 条 episode，得分 0.744 → 0.705。

论文由此得出“两类记忆互补且不可缺少”。但同一篇论文的超参分析（RQ3）显示，得分随 `k` 从 2 增至 10（条目总数 6 → 30）急剧上升，`k > 10` 后趋于平台。这意味着 30 条目已接近饱和区，而 20 条目和 10 条目分别落在上升段上。**去掉一个通道同时删去了 1/3 到 2/3 的检索预算，两个变量没有分离。**

MAGMA 的单图消融（Table 5）存在同类问题：`Causal Only` 0.590、`Temporal Only` 0.577、`Entity Only` 0.531，`Full MAGMA` 0.700。单图变体只在一张图上做 beam 遍历，候选集必然小于四图合并，而论文只报告了 Full MAGMA 的 3.37k tokens/query，没有报告单图变体的 token 预算。

MIRIX 和 Hindsight 则完全没有对各自的记忆划分做消融。MIRIX 的 Active Retrieval 对**六个组件各取 top-10**，即一次注入最多约 60 个条目再送入 system prompt；它与 RAG-500 等基线的对比因此同样不是等预算比较。Hindsight 在结论中称“this structure matters in practice”，但支撑证据只有端到端分数，且 LoCoMo 部分基线数值直接引自 Backboard 的公开榜单而非独立复现。

需要指出的是，等预算消融是可以做到的，SYNAPSE 就做了：其 recall 消融固定 `retrieval_topk = 30`，只改变三信号融合权重，`(1.0, 0, 0)` 纯向量的 multi-hop `Recall@30` 为 0.679，`(0.5, 0.3, 0.2)` 加入 activation 与 PageRank 后为 0.841。因为预算不变，这个结论可以归因到结构信号本身。其机制级消融（Fan Effect、Lateral Inhibition、Node Decay）同样只改打分动态、不改 Top-30，因此也是干净的。

**结论：分类结构可能有价值，但除 SYNAPSE 外，现有工作报告的证据不足以把收益归因到分类，而不是归因到检索了更多信息。**

### 1.3 检索路由：剪枝的单点失效与不剪枝的无收益

在分类之上做 query 路由，是另一种论证分类价值的方式，但它面临一个两难：

- **路由若真的剪枝**，则成为不可恢复的单点失效。MAGMA 的消融显示，移除 Adaptive Policy 造成全系统最大跌幅（Judge 0.700 → 0.637），说明路由质量主导了最终效果；反过来说，路由判断错误时被剪掉的分支不会在下游被找回。
- **路由若不剪枝**，则不产生收益，只增加成本。MIRIX 的 Meta Memory Manager 在写入侧路由，读取侧的 Active Retrieval 却是对六类各取 top-10 全部注入，本质上没有做选择。

而路由本身要额外付出一次 LLM 调用。MIRIX 的 Chat Agent 更进一步：先做一次跨六组件的粗检索拿摘要，再判断哪些组件值得深入检索并选择检索方法，最后综合作答——一次问答内多次模型往返。

有一个与之相关的现象值得记录：MemOS 在 PreFEval 上的对比中，注入 10 轮无关对话后，MIRIX 的 Preference Hallucination 从 9.5% 升至 72.0%，Personalized Response 从 37.7% 跌至 7.9%。这说明按类型组织并全量注入，在存在干扰内容时并不稳健。

### 1.4 动态聚类的现有尝试与其失效点

另一类工作放弃固定类型，改为按内容动态形成组织单元。它们与 FluxFold 的 subject 最为接近，因此需要逐一说明差别。

**EverMemOS 的 MemScene** 是增量语义聚类：新 MemCell 计算 embedding，在时间间隔满足 `tc - tlast(S) ≤ Δmax` 的候选 scene 中找质心 cosine 最高者，满足 `sim > τ` 且无 profile 冲突则并入并用滑动均值更新质心，否则新建 scene。这一设计有两处结构性弱点，且论文自己的数据就暴露了它们：

- **只能吸收，不能分裂。** scene 的质心是成员 embedding 的 running mean，成员越多质心越趋向平均，越难对新成员做出有区分度的判断，但没有任何机制在 scene 变大后重新划分它。
- **实测聚类几乎没有形成主题结构。** 在 LoCoMo（每个 conversation 约 71 个 MemCell）上，其 Table 12 报告平均 41.1 个 scene、平均规模 1.84 个 MemCell，Separation（intra 减 inter 相似度）仅 **+0.007**（intra 0.740 / inter 0.733）。也就是说，多数 scene 实质上就是单个 MemCell，且 scene 之间在向量空间上几乎不可区分。论文强调这是四种方法中唯一为正的 Separation，这一点成立，但正值本身极小。
- **时间门 `Δmax`（LoCoMo 7 天 / LongMemEval 30 天）会切断长周期主题。** 一个跨越数月反复出现的主题会被拆成若干互不相连的 scene。

**MemBox 的 Trace** 是跨 box 的事件轨迹：新 box 的每个 event 与既有 trace 的全部 event 计算最大 cosine，达到阈值（代码默认 `0.5`）者才进入 LLM 验证。它允许分叉与交叠，不是互斥分类树。论文在 Limitations 中明确承认其边界：event trace 适合活动、计划和事实进展，但“user preferences, interpersonal relations, affective states, stable personality traits, or evolving constraints”这些不以事件形式出现的长期记忆维度，当前 Trace Weaver 无法覆盖。

**CompassMem 的 Topic** 是在事件图之上的粗粒度层。论文描述的是在线分配加每 4 个 construction step 的周期性重聚类；当前实现则是对全部 `N*` 节点做一次批量 K-means，簇数默认 `max(2, min(n/5, 50))`。重聚类会整体重排组织单元，无法保证同一主题在两次聚类间保持稳定身份，也就无法围绕它做增量维护。

**ByteRover 的 Context Tree** 是 Domain → Topic → Subtopic → Entry 的层次结构，由写记忆的那个 LLM 通过 `curate()` 直接决定树的形状，条目间靠作者显式声明的 `@relation` 连边。这是唯一把组织权完全交给 LLM 的方案，但它的消融显示：在 LongMemEval-S 上移除 Relation Graph 只使总分下降 0.4 个百分点（92.8% → 92.4%），主要影响集中在 temporal reasoning（−2.2 pp），multi-session 反而略升。**显式声明的跨条目关系边，实测贡献很小。**

---

## 2. 关联发现：整理阶段没有 query 的结构性困难

### 2.1 现有整理机制全部被“新条目自身的相似度”门控

检索时有 query 引导，可以判断什么是相关的；整理时没有 query，系统唯一可用的廉价信号就是新记忆自身的向量。因此几乎所有工作的整理阶段，都用同一个模式：**以新记忆为 query 召回近邻，再把近邻交给 LLM 判断。**

| 系统 | 整理阶段的候选来源 | 门控 |
|---|---|---|
| A-MEM | ChromaDB 对新 note 的 dense 近邻 | 固定 5 个 |
| Mem0 | 每条候选事实召回旧记忆 | top-s = 10 |
| EverMemOS | 新 MemCell 与 scene 质心 | `sim > τ`（0.70 / 0.50）且时间窗内 |
| MemBox | 新 event 与 trace 内全部 event | 最大 cosine ≥ 0.5 才进入 LLM 验证 |
| Graphiti | 新 fact 的 hybrid 检索候选 | 检索 top-k |

这个模式有一个无法通过“换更好的 LLM”解决的上界：**LLM 再强，也只能在被送进 prompt 的候选里建立关联。** 如果一条真正相关的旧记忆与新记忆的向量相似度低于门槛，它根本不会成为候选。A-MEM 的情况最典型——它把 agency 放在记忆组织阶段，是这批工作中最强调“让 LLM 决定连接”的一个，但 LLM 每次只看到 5 个 dense 近邻。

### 2.2 为什么“相似度之外的关联”是实际需求

考虑两条记忆：`Mike went to the dental hospital to have a tooth implanted yesterday` 与 `Mike prefers alcoholic beverages`。二者在词面和向量上都不接近。但在“种植牙后短期内不宜饮酒”的常识下，它们存在实质关联：当用户请 Agent 推荐饮品时，遗漏这个关联会导致明显错误的建议。

这类关联的特征是：**关联性由外部常识或因果知识决定，而不由两条记忆的文本相似度决定。** 上述所有整理机制都无法发现它，因为两条记忆互相不会成为对方的候选。MemBox 的 Limitations 从另一个角度描述了同一问题：非事件型的约束（evolving constraints）无法被其 event trace 表达。

需要区分的是，SYNAPSE 和 CompassMem 处理的是**检索时**的类似问题——SYNAPSE 用 spreading activation 让能量沿 `query → concept → related concept → episode` 传播，CompassMem 用 LLM Explorer 沿 typed relation 逐步探索。二者都能召回与 query 表面不相似的证据，但前提是**图里已经存在那条边**。如果整理阶段就没有建立这条边，检索阶段再复杂的传播也无从利用。**关联发现的瓶颈在写入侧，而现有工作的补救几乎全部在读取侧。**

---

## 3. 检索：结构关联不等于 query 条件下的证据相关性

### 3.1 细粒度图记忆：清晰的组织形式与检索时的失配

以 Mem0<sup>g</sup> 和 Graphiti 为代表的一类工作，把原始经历拆成细粒度的 entity、relation 或 fact，用图保存它们之间的联系。与把整段历史作为文本块保存相比，这种结构便于围绕同一实体聚合跨时间信息、执行去重、冲突处理和时间更新。

但这些优势主要发生在添加与整理阶段。进入检索阶段后，系统需要判断的是“哪一条记忆能够回答当前 query”，而图中表达的是“哪些信息彼此存在结构关联”。**结构关联（association）并不等同于 query 条件下的证据相关性（query-conditioned relevance）。**

图记忆通常同时使用两类信号：`content-based relevance`（BM25 等 lexical 或 embedding cosine 等 semantic）直接比较 query 与节点内容；`structural relevance` 先依据 query 找 anchor，再沿邻接关系扩展，用 hop distance 或传播 activation 衡量接近程度。具体实现各异——Mem0<sup>g</sup> 同时使用 entity-centric graph retrieval 和 query-to-triplet semantic retrieval；Graphiti 支持 BM25、cosine、BFS 及多种 reranker，默认搜索主要是 BM25、cosine 与 RRF 的组合。因此问题不在某个具体打分公式，而在所有邻域扩展共同面对的限制：**图距离只说明候选与 anchor 在现有图中有多近，不保证它对当前 query 有多重要。**

这一限制在个人 Agent 场景中尤其明显。如果抽取策略反复把事实连接到 `user`、常见人物或高频主题，这些节点就会成为 hub，大量内容距离 `user` 只有一到两跳，hop distance 因而失去区分力。SYNAPSE 的消融量化了这一点：关闭 Fan Effect（不按节点出度稀释 activation）后，Open-Domain 从 25.9 跌至 16.8，平均 F1 从 40.5 跌至 36.1，论文的解释正是 common-entity hub 积累了过多能量并淹没了具体信号。

结果是，结构扩展通常只能负责把潜在相关的记忆放入候选集，最终仍需 lexical、semantic 或 cross-encoder relevance 筛选重排。这压制了扩散噪声，也使图容易退化为一种高成本的候选生成手段：结构召回的证据若与 query 缺少直接内容相似度，仍可能在最后排序中被过滤掉。

### 3.2 多跳问题：答案位于一条窄路径上

这种矛盾在 LoCoMo 的 multi-hop 和 open-domain 问题中最突出。此类问题的答案分散在多条细粒度记忆中，单条既不完整，也未必与 query 表面相似。

例如记忆中分别保存：`M1` 用户预订了 5 月 10 日的京都行程；`M2` 产品发布从 5 月 15 日提前到 5 月 10 日；`M3` 用户负责产品发布现场；`M4` 用户后来取消了京都行程。对于“用户为什么取消京都行程？”，内容检索容易召回含“取消”和“京都”的 `M4`，但 `M4` 本身没有答案。真正的解释需要沿“取消的行程 → 日期冲突 → 发布提前 → 用户必须到场”组合 `M1`、`M2`、`M3`。宽泛扩散会混入大量无关的旅行、工作和日程记忆；只保留与 query 最相似的 top-k 又会排除相似度较弱的 `M2`、`M3`。关键不是能否遍历更多节点，而是能否在每次分叉时判断哪条 relation 对尚未回答的部分仍有证据价值。

Mem0 的实验直接体现了这一局限：LoCoMo multi-hop 上 Mem0<sup>g</sup> 的 F1 为 24.32，**低于**不使用图的 Mem0 base 的 28.64。图中存在更多关联，并没有自然转化为更好的多跳检索。

### 3.3 主动图探索与延迟预算

CompassMem 引入了 query-conditioned 的主动探索：Planner 把 query 拆成 2–5 个 subgoals，多个 Explorer 从不同 topic 的候选事件出发，在图上逐步选择 `SKIP` / `EXPAND` / `ANSWER`，队列耗尽仍有缺口时 Planner 改写 query 再搜一轮。这显著改善了复杂问题的检索质量，但其 GPT-4o-mini 实验的 multi-hop F1 仍为 38.84，且代价是每题多次 LLM 往返。

把各系统报告的检索延迟并列，可以看出成本随读取路径上的 LLM 调用次数呈量级变化：

| 读取路径形态 | 系统与实测延迟 | 来源 |
|---|---|---|
| 纯向量 | Mem0 search p50 0.148 s / p95 0.200 s | Mem0 Table 2 |
| 纯向量（服务化） | MemOS-1031 mean 440 ms / P99 777 ms（10 QPS） | MemOS Table 7 |
| 图遍历，无 LLM | Mem0<sup>g</sup> 0.476 / 0.657 s；Zep 0.513 / 0.778 s；A-Mem 0.668 / 1.485 s | Mem0 Table 2 |
| 单次 LLM 路由/规划 | MAGMA 平均 1.47 s；ByteRover cold p50 1.2 s（LoCoMo）/ 1.6 s（LongMemEval-S） | MAGMA Table 3；ByteRover Table 5 |
| 少量 LLM 往返 | EverMemOS 端到端 3–5 s（默认 2 次调用，触发改写时 3–4 次） | EverMemOS 附录成本表 |
| Agentic 循环 | ByteRover Tier 4 8–15 s；CompassMem 平均 20.87 s / multi-hop 24.61 s；MemoryOS 平均 32.68 s；LangMem search p95 59.82 s | 各自论文 |

这些数字来自不同 harness、不同硬件和不同后端，不能直接横向比较绝对值；但量级分组是稳健的，可以据此校正一个常见的粗糙说法。**并非“检索路径上出现 LLM 就必然导致极高延迟”：一次路由或规划调用大致增加 1 秒量级。真正不可接受的是逐跳或迭代式的 LLM 决策，它把延迟推到 10–30 秒量级。** 对于期望在亚秒到数秒内响应、且主 Agent 一轮内可能多次调用记忆工具的同步产品，后者很难作为默认检索路径。

即便是单次调用，代价也不只是那 1 秒：它同时引入了 §1.3 描述的路由单点失效风险，以及一次额外的模型故障面。ByteRover 在 Limitations 中承认得很直接——当查询未命中缓存和索引而升级到 Tier 3–4 时，“ByteRover requires an LLM call that a vector similarity search does not”，其整个设计建立在“Agent 反复问一小组问题的变体、缓存能吸收大部分负载”这一假设上；假设不成立时延迟就会上升。

### 3.4 小结

上述工作的演进揭示了一项尚未解决的张力：细粒度结构擅长保存关系，却难以低成本地把这些关系转化为 query-conditioned evidence selection。宽泛扩展提高召回但带来 hub noise 和候选膨胀；强剪枝和 content-based reranking 控制成本但遗漏语义不相似的深层证据；逐步用 LLM 选路提高准确率但引入难以接受的延迟。

因此需要回答的不是“是否使用图”，而是：**如何在不展开整个邻域、也不依赖逐跳 LLM 调用的前提下，让记忆之间的结构关系真正参与当前 query 的证据选择。**

---

## 4. 演化与冲突：provenance 的存与用

### 4.1 存了不用，或根本不存

记忆会重复、冲突、被纠正、发生状态变化。系统据以做出“保留 / 修改 / 失效”判断的输入是什么，决定了这些决策的可靠性。按此考察参考工作，可以分为三种情况：

**根本不保留来源。** Mem0 的向量库 payload 只有抽取后的记忆句 `data`；`~/.mem0/history.db` 记录 ADD/UPDATE/DELETE 审计与最近消息，但不是每条记忆到其来源片段的可寻址映射。SimpleMem 的对话只作滑动窗口缓冲、不长期存储，`MemoryEntry` 只保留 `lossless_restatement`。A-MEM 的 note 内容本身就是唯一表示。Hindsight 的 `memory_units` 只有叙事 `text` 和 `proof_count`。

**保留了来源，但冲突判断不读取它。** 这一类更值得注意，因为它说明保存 provenance 与使用 provenance 是两件事：

- Graphiti 保存了不可损的 Episodic 层原文，但其边冲突解决 prompt（`graphiti_core/prompts/dedupe_edges.py::resolve_edge`）的输入只有 `existing_edges`、`edge_invalidation_candidates` 和 `new_edge` 三组**事实字符串**，不包含任何 episode 正文。模型判断“Alice works at Acme Corp as a software engineer”是否被“…as a senior engineer”推翻时，看不到这两句话各自是在什么语境下说出的。
- Mem0 的 `DEFAULT_UPDATE_MEMORY_PROMPT` 同样只接收 `Old Memory` 的文本列表与 `Retrieved facts`，靠这两组压缩后的句子决定 ADD / UPDATE / DELETE / NONE。
- Nemori 的 episode 保存了 `source_messages`，SYNAPSE 保留 episodic node，MAGMA 在 `attributes.raw_content` 中保留原文；但这些原文的用途是作为检索结果送入回答 prompt，不是作为整理阶段的裁决依据。

**保留的是引用而非内容。** CompassMem 的图文件只保存 `utterance_refs` 和摘要化的 `texts`，回答时需要 `ConversationManager` 从 `qa_data_path` 指向的原始 LoCoMo / NarrativeQA 文件重新解析原句——graph JSON 不是自包含的记忆包，脱离数据集目录就无法还原来源。

结果是，绝大多数系统在做记忆演化决策时，看到的只是自己此前压缩的结果。当两条压缩后的记忆表面矛盾时，系统只有两个选择：按时间新旧覆盖，或者保留矛盾。它无法回答那个真正重要的问题——**这两条记忆是真的矛盾，还是各自在不同条件下成立？**

用一个具体例子说明区别。记忆中同时存在“用户偏好在实现细节不明确时被追问”和“用户希望工具更自主地完成实现”。仅看这两句话，只能判定为冲突。回到来源 episode 才能发现：前者出自一个科研项目的对话，后者出自一个个人小工具的对话。正确的演化结果不是让其中一条覆盖另一条，而是把两者各自补全为带条件的记忆。缺少可读取的 provenance，这个判断做不出来。

### 4.2 单一用户画像的粒度问题

与之相关的是“用户画像”这一常见形态。MIRIX 的 Core Memory 分 `persona` 与 `human` 两个 block，**始终出现在 system prompt 中**，容量超过 90% 时触发压缩重写；EverMemOS 维护单一 User Profile（explicit facts 与 implicit traits 两个字段），由 scene summary 在线更新；MemOS 有 PreferenceTextMemory 区分显式与隐式偏好。

这类设计有两个耦合在一起的性质：**粒度粗**（一个用户一份画像）与**常驻**（不经 query 条件筛选就进入上下文，且优先级高）。二者结合会产生一种典型故障：在项目 A 中形成的工作方式偏好，会被无差别地应用到与项目 A 无关的任务上；关于用户正在做项目 A 的记忆，会让 Agent 在回答无关技术问题时反复关联到项目 A。这不是记忆错误，而是**记忆被用错了场合**——一个作用域问题，而非召回问题。

需要诚实地指出：**目前没有 benchmark 会惩罚这种失败。** LoCoMo、LongMemEval 的问题本身就以“答案存在于历史中”为前提，PersonaMem-v2 和 PreFEval 度量的是偏好是否被正确使用（MemOS 报告的 Preference Unaware 4.6%、MIRIX 在 10 轮干扰下 Preference Hallucination 72.0%），衡量的都是**该用而没用**或**用错内容**，没有一项衡量**不该用却用了**。因此这条局限有清晰的机制解释和真实的使用反馈，但缺少公开的量化证据。任何声称解决它的设计，都需要先构造相应的评测。

---

## 5. 现有评测无法暴露的问题

除了上一节末尾的作用域问题，还有一个更基础的方法论限制值得单独记录，因为它直接决定了 FluxFold 应该怎样做实验。

**写入期的组织机制无法通过 query-time ablation 评估。** ByteRover 对此有明确说明：Adaptive Knowledge Lifecycle、curation 反馈环和 escalated compression 三项机制“operate exclusively during the write path”，由于消融时 Context Tree 保持不变，它们的贡献“cannot be isolated through query-time ablation on a static benchmark”，隔离它们需要在退化条件下重新 curate。

这个限制适用于本文讨论的大部分组织机制——聚类、分裂、关联建链、冲突消解，全都发生在写入侧。而多数论文的消融是在固定记忆库上关闭某个读取通道（Nemori 的 `w/o e`、MAGMA 的单图变体、Hindsight 的无消融）。**这解释了为什么“组织结构有价值”这一论断在文献中普遍缺少干净证据：不是结论错误，而是主流实验设计测不到它。** 要评估写入期机制，必须在关闭该机制的条件下**重建整个 memory space**，再用同一组 query 和同一检索预算比较。

---

## 6. FluxFold 的设计取舍

### 6.1 总体判断：把 LLM 成本全部前移到写入路径

FluxFold 的核心工程判断是：**读取路径不使用 LLM，全部 LLM 成本前移到写入与整理阶段。**

公开 `search(query)` 只做精确向量检索——两个通道分别计算 query 与 active subject name embedding、active memory content embedding 的 cosine 相似度，按 top-k 和最低相似度阈值筛选，合并后只按真实 link 去重整理，不计算融合分数、不重排、不做第二轮筛选，也不改写 query。按 §3.3 的量级分组，这把检索成本放在“纯向量”一档，同时避免了 §1.3 的路由单点失效。

相应地，写入阶段承担五类 LLM 调用：memory extraction、批次 Subject linking、Subject review、Subject split、link-local Subject summary refresh。这个取舍的合理性依赖一个前提：**在写入时把关系整理清楚，检索时就不必用 LLM 临时重建关系。** 这个前提是否成立，需要实验验证；§6.6 列出了它可能不成立的情形。

### 6.2 设计一：subject 作为有界、可自我具体化的动态组织单元

`subject` 是围绕人物、项目、话题、事件或其他可独立组织范围的一组记忆，通过 `subject_memory_links` 与 memory 多对多关联。它与前述各类组织单元的关键差别在于**规模有界且会自我具体化**：

- **有界维护。** 一个 subject 自上次成功 review 或 split 后每新增 8 条带有新 active link 的 memory，触发一次 review。review 读取该 subject 当前**全部** active memories（而非仅新增的 8 条），可以保留、修改和全局退役记忆，并整体重写 summary。这正是 §1.1 中固定分类无法提供的操作：一个规模可控、语义内聚、可被完整复核的子集。
- **达到阈值后分裂。** active memory 数达到 24 或 latest content 总字符数达到 8,000 时尝试 split，结果只能是 `full_split`（创建 2–5 个更具体的新 subject，原 subject 退役）、`partial_split`（拆出 1–4 个新 subject，原 subject 保留 ID、name 和 active 状态）或 `defer_split`（当前记忆无法形成有意义的分组，记录 warning，下次再有 link 时重试）。分组依据是“未来是否需要独立检索、更新和增长”，而不是平均分配数量；结果 name 必须保留原主体锚点并表达具体领域、项目模块、事件阶段或人物关系，不允许 `Other`、`Misc` 这类没有语义边界的名称。

这使得 subject 的语义随记忆积累而演进：初期可能接近实体（`Mike`），随后分化出 `Mike's dietary preferences`、`用户在项目 A 中的代码风格偏好` 这类更具体的组织单元。

**与最接近的参考工作的差别：**

| 参考单元 | 组织方式 | 与 subject 的关键差别 |
|---|---|---|
| EverMemOS MemScene | 质心相似度 + 时间窗 `Δmax` 增量吸收 | 只吸收不分裂，质心随成员增多而钝化；`Δmax` 切断长周期主题；LoCoMo 实测平均规模 1.84、Separation +0.007。subject 无时间门、无质心，且到达阈值必须尝试分裂 |
| MemBox Trace | event 相似度 + LLM 验证的事件轨迹 | 论文自述只覆盖事件型内容，不覆盖偏好、关系、人格、持续约束。subject 不限定内容形态 |
| CompassMem Topic | 对全部节点批量 K-means | 重聚类破坏组织单元的稳定身份，无法围绕它做增量维护。subject 有稳定 ID，partial split 明确保留原 ID 与 name |
| ByteRover Context Tree | LLM 直接 curate 的 Domain→Topic→Subtopic 层次 | 维护一棵持久层次树，需要保证全局一致性。FluxFold 不保存 subject 层级、不主动维护包含关系；包含关系只在 LLM 单次判断“只 link 到更具体的那个”时使用，不落库、不校验 |
| 图记忆的 entity node | 抽取时按实体命名 | entity 的语义在系统生命周期内固定，度数无上界；subject 的语义会具体化，规模有软阈值并触发整理 |

**subject 在检索中的桥接作用**也与 hub 节点不同。`search` 的两个通道互相附带关联实体：Subject 通道召回最多 5 个 subject，每个附带其 active memories 中与 query 最相似的 1 条；Memory 通道召回最多 15 条 memory，每条附带其 active subjects 中 name embedding 与 query 最相似的 1 个。结果按 subject 去重分组，只有 Subject 通道直接命中的最多 5 个 subject 展示 summary；Memory 通道额外带入的 subject 只展示 name。因为 subject name 是较短、较概括的文本，两条互相不相似的记忆可以同时与同一个 subject name 保持较高相似度，subject 因而成为它们之间的桥。与图 hub 的区别在于：subject 的成员集合是**有界且被整理过的**（review 或 link-local refresh 重写 summary、split 拆分过大的 subject），而 hub 节点的邻居是无界累积的——§3.1 中 SYNAPSE 需要 Fan Effect 抑制的正是后者。

不过必须准确描述这个机制的能力边界：一次 search 只做一跳附带、每个候选只附带 1 个关联实体。§2.2 的种牙/饮酒例子中，真正承载跨记忆关联的不是这一跳附带，而是 **subject summary**——review，或新 memory 链接已有 subject 后的局部 refresh，会基于该 subject 当时的全部 active memories 重新生成 summary，跨记忆的约束关系在这一步被写入 summary 文本。这也意味着该能力依赖 summary 质量，而不是检索结构本身。

FluxFold 有意采用**弱一致性**：memory version、provenance 和 active link 是事实源，subject summary 只是写入期生成的派生检索文本。系统不持久化 dirty 状态，也不扫描全空间补齐 summary；共享 memory 因另一个 subject 的 review 发生变化时，其他 summary 可以继续保留旧表述，直到未来新 memory 再次链接该 subject，或该 subject 自身 review/split。这个取舍避免把每次局部写入扩散成全局 LLM 维护，但意味着公开 search 在收敛前可能看到陈旧 summary。

### 6.3 设计二：写入期的主动关联检索与双通道候选召回

针对 §2 的问题，FluxFold 在写入阶段做两件事：

**双通道被动召回。** 以新 memory content 为 query，Subject 通道按 subject name embedding 取前 8 个（相似度 ≥ 0.25），每个附带 1 条最相似的关联 memory；Memory 通道取前 16 条 memory（相似度 ≥ 0.35），每条附带 1 个最相似的关联 subject。两个通道完成后按真实 subject–memory 关系去重并组织成 subject groups；一个 linking 请求跨全部新 memories、以及适用时的主动检索结果，全局只展示相似度最高的 5 个不同 subject summaries，每份 summary 只出现一次。不计算融合分数、不建更大的中间候选池。相比 A-MEM 的 5 个 dense 近邻，这让 LLM 同时看到“与新记忆相似的记忆”和“与新记忆相似的组织单元及其代表记忆”两种视角，同时避免重复展示同一 subject summary。

**一次可选的主动关联检索。** 开启后，LLM 在做 linking 决策前最多调用一次 `association_search(query)`，且 prompt 明确要求该 query **表达可能的影响方向，而不是复述新记忆**。这是与 §2.1 表格中所有机制的实质差别：其余系统的候选集完全由新条目自身的向量决定，FluxFold 允许模型主动提出一个不同的检索方向。种牙那条记忆写入时，模型可以检索“饮食限制 / 术后禁忌”而不是“牙科就诊”，从而把饮酒偏好拉进候选。

**批次一次性决策。** 程序为本 episode 的全部新 memory 分配临时 `memory_ref`；各 memory 的候选可分别召回，但 LLM 必须在一个最终输出中给出整个批次的全部 memory–subject links 以及全部待新建 subject。同批 memory 不是依次写入，任何新 subject 都在完整批次校验通过后才取得正式 ID，整批与 memory、provenance、extraction completion 原子提交。这避免了逐条写入时“先写的记忆看不到后写的记忆”造成的组织不一致。

每条 link 记录 `direct` 或 `contextual` 依据：`direct` 表示 memory 直接描述该 subject 或其核心事实属于该 subject；`contextual` 表示 memory 不直接描述它，但会具体影响、约束、更新或解释该 subject 下的信息，遗漏它可能实质损害未来回答。`contextual` 不表示较低优先级，两类 link 都参与 review、summary 和 search，第一版不据此调整检索分数。跨领域的影响关系正是通过 `contextual` link 表达的。

需要说明的是，`association_search` 目前上限为 1 次调用，且 design.md 已将“对关闭与开启主动关联性检索做消融，评估关联召回率、错误 link 和调用成本”列为待验证项。按 §5 的结论，这项消融必须通过**重建 memory space** 进行，不能在固定记忆库上关闭读取通道。

### 6.4 设计三：episode 级 provenance 作为演化依据

针对 §4.1，FluxFold 把 provenance 做成正式数据结构并让它进入决策：

- `memory_version_provenance` 是 memory version 与来源 episode 的多对多关联表。一个 memory version 由 1–6 个不同 episode 支持；prompt 与结构化校验使用相同上限，超过时不得截断，无法由至多 6 个来源准确支持的合并不得执行。
- memory 的稳定身份（`memory_id`）与内容历史（`memory_versions`）分离，历史版本各自保留自己的 provenance。修改记忆创建新版本，旧版本 `is_latest` 置 0。
- **review 可以按需读取来源。** 仅当 content 与元数据不足以解决重复、冲突、纠正、状态变化或信息归属时，首轮可以输出 `provenance_request`，一次最多指定 8 个 memory ID，系统返回这些记忆当前 provenance 涉及的全部 episodes；获得来源后必须输出最终结果，不能再次请求。这是与 Graphiti、Mem0 的直接差别——后两者的冲突判断 prompt 里没有任何来源正文的位置。
- **冲突不被默认覆盖。** review 不能默认新事实覆盖旧事实，也不能仅因时间较早退役某条记忆；只允许拼接共同描述同一个、不可独立更新事实的 memories，不同但相关、可以分别变化的事实继续分开；无法解决的冲突应保留，并在 summary 中准确表达不确定性。
- **退役是逻辑删除。** `retired` 记忆保留正文、provenance、原 subject links 和 operation history，不再参与任何召回、审核、分裂、summary 或公开 search。`domain_operations` 与 `domain_operation_effects` 是 append-only 的结构化审计记录，episode、历史版本和 domain operation 用约束或 trigger 防止修改和删除。

这使 §4.1 末尾的例子有了可执行路径：两条表面冲突的偏好记忆触发 review，模型判断仅凭文本无法裁决，请求 provenance，读到两条来源 episode 分别属于科研项目与个人工具的对话，据此把两条记忆各自改写为带条件的表述，并把新的 provenance 集合一并写入。整个过程留下可追溯的 operation 记录。

### 6.5 与用户画像方案的差别

FluxFold 不维护常驻的用户画像。它对 §4.2 作用域问题的处理是机制性的，而不是语义理解性的：

1. **按需召回而非常驻。** subject summary 与 memory content 只在 query 与之相似度超过阈值时才进入结果，不占据固定的 system prompt 位置。
2. **分裂降低了错误召回的相似度。** 这是 split 机制的一个直接副效应：一个名为“用户”的粗粒度 subject 与几乎任何查询都有中等相似度；而“用户在项目 A 中的代码风格偏好”这样的名称，对“推荐一款饮品”这类查询的相似度会低于 0.25 的阈值，因而不会被召回。**记忆越具体，被误召回的概率越低。**

必须明确：这不等于系统“理解”了当前场景是否适用某条偏好，它只是让不适用的记忆在向量空间上离当前 query 更远。效果的强弱完全取决于 subject name 的具体化程度和阈值设置，而这两者目前都未经校准。

### 6.6 尚未解决的问题与必须验证的假设

按 §5 的方法论要求，以下每一项都需要重建 memory space 后比较，而不是在固定记忆库上关闭读取通道。

**检索侧的已知缺口。**

- 第一版所有通道纯向量，不使用 BM25、关键词、融合或重排。而 Zep、Hindsight、EverMemOS 都用 dense + BM25 + RRF，SimpleMem 另有符号层做时间与实体过滤。**精确匹配类查询（人名、日期、数值）和 temporal reasoning 是纯向量的传统弱项**，LoCoMo 与 LongMemEval 都有相应题型。SYNAPSE 的 BM25 lexical trigger 与 dense trigger 取并集也说明了这一点。
- 相似度阈值 0.25 / 0.35 与 top-k 5 / 15 依赖具体 embedding model，尚未针对所选模型校准。阈值偏高会静默丢弃相关结果，偏低则失去筛选作用。
- 不设最终返回字符上限。新 memory 链接已有 subject 后，会对本次局部目标中未经过 review/split 的 subject 按全部 active memories 重写 summary，并受 2,000 字符硬上限约束；但被召回的 5 个 summary 与最多 20 条 memory 仍可能占用较多返回预算。

**组织侧的已知缺口。**

- **无 link 回填。** 新建 subject 只建立当前 episode 组织批次中由 LLM 选定的 memory links，不回填更早 episode 的 memories。一个在交互后期才成立的 subject，永远不会关联到它本应涵盖的早期记忆。
- **summary 成本与陈旧窗口。** 每次 link 后都对本次命中的已有 subject 中未经过 review/split 者执行一次全量 summary refresh；按 subject 去重，但每个目标需要独立 LLM 调用。系统不因共享 memory 被修改或退役而刷新其他 subject，也不保留失败待办，因此避免了全空间维护成本，却允许 summary 在没有后续相关写入时长期陈旧。需要通过实验同时衡量调用成本、summary 质量和陈旧信息对检索的影响。
- **软阈值不设上限。** 实验版不限制单个 subject 最终关联的 memory 数量或总字符数，`defer_split` 是合法结果。理论上一个语义确实无法细分的 subject 可以无限增长，而 review 每 8 条新记忆就要读取它的**全部** active memories，成本随规模线性上升。
- **分裂是否会被触发，需要实测。** EverMemOS 在 LoCoMo（每 conversation 约 71 个 MemCell）上聚出平均规模 1.84 的 scene，这是一个警示：在 benchmark 规模的数据上，组织单元仍可能达不到 24 条 memory / 8,000 字符的分裂阈值。若 split 在两个数据集上极少触发，则“自我具体化”这一核心机制在实验中不会被检验到，需要补充设计针对性的评测数据或继续调整阈值。
- **provenance 深度受限。** 一次 review 只允许一次 provenance request（最多 8 个 memory ID），一个 memory version 最多关联 6 个 episode。跨越很多轮次逐步演化的长期冲突，可能超出这个预算。

**尚无评测支撑的部分。** §4.2 的作用域问题（记忆被用在不该用的场合）是 FluxFold 的重要动机之一，但如该节所述，现有公开 benchmark 都不度量它。**在构造出相应评测之前，不应把“解决了用户画像污染上下文的问题”作为已验证的结论陈述。**

---

## 7. 参考材料

- [A-MEM: Agentic Memory for LLM Agents](A-mem/3_A-MEM.pdf)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](mem0/3_Mem0.pdf)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](graphiti/2_Zep.pdf)
- [MemOS: A Memory OS for AI System](MemOS/4_MemOS.pdf)
- [MIRIX: Multi-Agent Memory System for LLM-Based Agents](MIRIX/5_MIRIX.pdf)
- [Nemori: Self-Organizing Agent Memory](nemori/7_Nemori.pdf)
- [Hindsight Technical Report](hindsight/8_Hindsight.pdf)
- [EverMemOS](EverOS/9_EverMemOS.pdf)
- [SimpleMem: Efficient Lifelong Memory for LLM Agents](SimpleMem/10_SimpleMem.pdf)
- [SYNAPSE](synapse/12_SYNAPSE.pdf)
- [MAGMA: Multi-Graph Agentic Memory Architecture](MAGMA/12_MAGMA.pdf)
- [Membox](Membox/14_Membox.pdf)
- [ByteRover](byterover-cli/16_ByteRover.pdf)
- [Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning](https://arxiv.org/abs/2601.04726)（CompassMem）
- Graphiti 冲突消解 prompt：`graphiti/graphiti_core/prompts/dedupe_edges.py`
- Mem0 记忆更新 prompt：`mem0/mem0/configs/prompts.py`
- SYNAPSE 等预算 recall 消融：`synapse/RESULTS.md`、`synapse/results/recall_ablation.json`
- FluxFold 设计：`FluxFold/design.md`、`FluxFold/design_detailed.md`
