# FluxFold Detailed Design

本文档是 FluxFold Memory Engine 第一版实现的完整设计来源，同时描述行为、数据表示、
持久化边界、配置数值和一致性要求。项目级交付形态、工具链与发布决策仍由 `design.md`
记录；Memory Engine 的实现不得依赖 `design.md` 中的重复描述或讨论草稿。

第一版分两个阶段：

1. **实验版设计**：实现 benchmark 所需的记忆添加、组织、维护和检索。
2. **正式版补充设计**：在实验版核心之上增加真实 Agent 交互的采集、缓冲、分段和恢复。

## 1. 第一版实验版设计

### 1.1 数据表示、存储与运行基础

#### 1.1.1 存储边界

- SQLite 是实验版正式数据和关系的唯一事实源。
- 数据集 session 作为已经划定边界的 `situational episode` 直接进入记忆提取流程，
不经过 durable inbox 或 situation boundary detection。
- SQLite 保存 episode、memory unit、provenance、subject、subject–memory link、
embedding、处理状态和 domain operation。
- memory、subject 和 provenance 是正式数据；embedding 是可从对应文本重新生成的
检索数据。
- memory latest version、active subject–memory link 和 provenance 是事务性事实；subject
summary 是由这些事实派生的检索摘要。每次 episode 的 add operation 持久化本次必须完成的
summary refresh targets，并分别记录完成状态。结构性写入已提交但 refresh 尚未全部完成时，
summary 可以暂时为空或过期；安全重放只重试仍未完成且仍 active 的 targets，不做全空间扫描。
共享 memory 被其他 subject 的 review 修改或退役时，除被 review 的 subject 本身外，不把
链接该 memory 的其他 subjects 自动加入本 episode 的 refresh targets。
- 实验版使用 SQLite 加 NumPy 完成精确向量扫描，不使用图数据库、向量数据库或 ANN
索引。
- Markdown 和 benchmark 输出是数据库状态的导出结果，不是可写事实源。

FluxFold 当前的关系是以 subject–memory 多对多关联和 memory–episode provenance 为主的
浅层关系。它们使用外键和关联表表达，不需要图数据库。

#### 1.1.2 SQLite 通用约定

- 内部 ID 使用 `TEXT` UUID；顺序由显式 sequence 字段表达，不依赖 UUID 排序。
- 系统时间使用 UTC Unix milliseconds，以 SQLite `INTEGER` 保存；来源时区单独保存。
- JSON 使用 `TEXT` 保存，并使用 `CHECK(json_valid(...))` 验证。
- 布尔值使用 `INTEGER`，并限制为 `0` 或 `1`。
- 所有文本长度都计算确定性预处理后字符串的 Unicode code point 数，并简称为“字符数”；
不使用估算 token 数或 UTF-8 字节数。应用层使用同一规则校验 message、memory content、
subject name、summary、query 及其他文本的长度、容量阈值、裁剪位置和
输出上限。表中保存 `char_count` 时，事务提交前重新计算并拒绝不一致的值。
- embedding 使用 little-endian、连续的 `float32 BLOB` 保存。
- 数据表优先使用 SQLite `STRICT` table。
- 数据库启用 `PRAGMA foreign_keys = ON` 和 WAL。
- 正式数据的外键默认使用 `ON DELETE RESTRICT`。退出 active 集合通过状态或关系结束时间
表达，不依赖级联物理删除。
- episode、历史版本和 domain operation 使用约束或 trigger 防止修改和删除。



#### 1.1.3 数据关系总览

```text
memory_spaces
├── situational_episodes
│   └── episode_blocks
├── episode_extractions
├── memory_units
│   └── memory_versions
│       ├── memory_version_provenance ──> situational_episodes
│       └── memory_embeddings
└── subjects
    ├── subject_embeddings
    └── subject_memory_links ──> memory_units

embedding_model_signatures
domain_operations
└── domain_operation_effects
```



#### 1.1.4 Memory space 与来源 episode



##### `memory_spaces`

`memory space` 是共享记忆组织和检索范围。实验版为一次独立 benchmark 运行或测试样例
建立一个 space，避免不同运行相互污染。


| 字段                | 含义              |
| ----------------- | --------------- |
| `memory_space_id` | memory space ID |
| `space_key`       | 调用方提供的稳定标识      |
| `created_at`      | 创建时间            |




##### `situational_episodes`

实验版将一个数据集 session 规范化为一个不可变 episode。


| 字段                                      | 含义                                                         |
| --------------------------------------- | ---------------------------------------------------------- |
| `episode_id`                            | episode ID                                                 |
| `memory_space_id`                       | 所属 memory space                                            |
| `source_type`                           | 数据来源，例如 `longmemeval` 或 `locomo_refined`                   |
| `source_key`                            | 数据集内 session 的稳定标识                                         |
| `source_sequence`                       | dataset session 在当前 memory space 内的确定性来源顺序；正式版 episode 可为空 |
| `payload_version`                       | episode block schema 版本                                    |
| `source_started_at` / `source_ended_at` | 已知的来源时间范围，可为空                                              |
| `source_timezone`                       | 解释来源自然时间的时区，可为空                                            |
| `content_hash`                          | 规范化 episode payload 的 SHA-256                              |
| `created_at`                            | episode 持久化时间                                              |


来源字段到 episode 时间和 message 的映射由 1.2.1 的 dataset profile 定义。数据集没有
逐消息时间时，block `observed_at` 保持 `NULL`，不得把 session 时间伪装成每条 message 的
精确发生时间。

同一来源 session 在一个 memory space 中只能写入一次：

```text
UNIQUE(memory_space_id, source_type, source_key)
UNIQUE(memory_space_id, source_sequence)
```

实验版 dataset episode 的 `source_sequence` 必须非空；正式版 episode 可以为 `NULL`，SQLite
允许同一 unique constraint 中存在多行 `NULL`。

##### `episode_blocks`

`episode_blocks` 按来源顺序保存 user message 和 assistant message。第一版的 tool call、
tool arguments 和 tool result 均不进入 Memory Engine，也不进入 episode 或 provenance。


| 字段                                 | 含义                                   |
| ---------------------------------- | ------------------------------------ |
| `block_id`                         | 稳定 block ID                          |
| `episode_id`                       | 所属 episode                           |
| `sequence_no`                      | episode 内顺序                          |
| `role`                             | `user` 或 `assistant`                 |
| `message_phase`                    | assistant message 是否为 `final`，其他情况为空 |
| `speaker_id` / `speaker_name`      | 多参与者数据中的身份，可为空                       |
| `content`                          | message 正文                           |
| `observed_at`                      | 来源时间，可为空                             |
| `metadata_json`                    | source-specific metadata，可为空         |
| `preprocessor_version`             | 产生当前内容的确定性预处理版本，可为空                  |
| `is_truncated` / `truncation_json` | 裁剪字段、原始与删除字符数、原内容 hash 和策略，可为空       |


协议角色与现实参与者分开表示，以同时支持 Agent transcript 和多参与者 benchmark 数据。

#### 1.1.5 Memory unit、版本与 provenance



##### `memory_units`

所有长期记忆统一使用无子类型的 `memory unit` 表示。


| 字段                        | 含义                         |
| ------------------------- | -------------------------- |
| `memory_id`               | memory unit 的稳定身份          |
| `memory_space_id`         | 所属 memory space            |
| `lifecycle_status`        | `active` 或 `retired`       |
| `latest_source_at`        | 当前 provenance 中最新的来源时间，可为空 |
| `created_at`              | memory 首次创建的系统时间           |
| `updated_at`              | memory 最近一次正式修改的系统时间       |
| `retired_at`              | 退役时间，可为空                   |
| `retired_by_operation_id` | 产生退役的 operation，可为空        |


`created_at` 和 `updated_at` 是 FluxFold 的系统时间。`latest_source_at` 是 provenance
派生的物化字段，表示这条记忆最近一次在来源对话中被观察或提及的时间，用于以后计算
时序衰减等新近性信号；它不表示 memory 的系统写入时间或 content 所描述事件的发生时间。

每个来源 episode 的时间锚点优先取 `source_ended_at`，缺失时取 `source_started_at`；
`latest_source_at` 是当前 latest memory version 的全部 provenance 时间锚点中的最大值。
没有可用来源时间时保持 `NULL`，不得使用 `created_at` 补造。provenance 被完整替换时必须
重新计算该字段，因此移除最新来源可以使它变早。该时间戳是持久化检索属性，不保存随
查询时间变化的 recency score。

`retired` 是全局逻辑删除状态。退役 memory 保留正文、provenance、原 subject links 和
operation history，但不再参与任何候选召回、审核、分裂、summary 生成或公开 search。

##### `memory_versions`

memory ID 表示稳定身份，version 保存该身份的内容历史。版本历史用于审核、benchmark
诊断和错误恢复，不改变公开 API 以 `memory_id` 操作记忆的语义。


| 字段                        | 含义                    |
| ------------------------- | --------------------- |
| `memory_version_id`       | memory version ID     |
| `memory_id`               | 所属 memory unit        |
| `version_no`              | 从 1 开始递增的版本号          |
| `is_latest`               | 是否为该 memory 的最新版本     |
| `content`                 | 自包含的正式记忆正文            |
| `content_hash`            | content 的 SHA-256     |
| `char_count`              | content 的 Unicode 字符数 |
| `created_by_operation_id` | 创建该版本的 operation      |
| `created_at`              | 创建时间                  |


每个 memory 必须恰好有一个 latest version。修改 memory 时创建新版本并将旧版本的
`is_latest` 设为 `0`；退役 memory 仍保留最后版本为 latest，但 lifecycle 使它退出正常
路径。旧版本不使用 `retired` 状态，以免与 memory 的全局退役语义混淆。

记忆的状态、时间、条件、信息归属和不确定性以自包含 content 为事实来源。第一版不建立
按记忆类型区分的时间子表，也不使用系统更新时间补造事件时间。审核所需的来源时间可由
provenance 关联的 episode 获得。

##### `memory_version_provenance`

`memory_version_provenance` 是 memory version 与来源 episode 之间的多对多关联表，只定位
到 episode 级别。一个 memory version 由一个或多个 episodes 支持；一个 episode 可以
同时支持零到多个 memory versions。`memory unit` 的当前 provenance 指其 latest version
的关联集合，历史版本分别保留各自的 provenance。

```text
PRIMARY KEY(memory_version_id, episode_id)
```

初次提取的 memory 只指向当前 episode。审核修改 provenance 时，LLM 提供的是新版本的
完整来源集合；系统验证所有 episode 有效且属于同一 memory space。overlap context 不得
成为本次新记忆的 provenance。一个 memory version 的 provenance 必须包含 1--6 个不同
episode；prompt 与结构化校验使用相同上限，超过时不得截断。provenance 和 episode 来源
时间是 `latest_source_at` 的事实来源。

#### 1.1.6 Subject 与 subject–memory link



##### `subjects`

`subject` 是动态维护、可分裂的记忆组织单元。


| 字段                          | 含义                                    |
| --------------------------- | ------------------------------------- |
| `subject_id`                | subject ID                            |
| `memory_space_id`           | 所属 memory space                       |
| `name`                      | 简短且可独立理解的名称                           |
| `summary`                   | 当前 summary；首次 refresh 前可为空             |
| `lifecycle_status`          | `active` 或 `retired`                  |
| `new_memory_count`          | 上次成功审核或分裂后新增的 memory 数量               |
| `summary_revision`          | summary 每次整体重写时递增的 CAS version          |
| `created_at` / `updated_at` | 系统时间                                  |
| `retired_at`                | subject 被替换的时间，可为空                    |
| `retired_by_operation_id`   | 产生替换的 subject split operation，可为空     |


subject name 在一次 subject 生命周期内保持不变。Subject review 只修改或退役 memories；
Subject split 可以完整替换原 subject、从中拆出若干更具体的 subjects 并保留原 subject，
或者明确推迟本次分裂。只有完整替换才退役原 subject；partial split 保留其 ID、name 和旧
summary，直到后续 summary refresh 整体重写。

link、review 和 split 都不读取、生成或改写 summary。link 或 split 新建 subject 时只写入
name，`summary` 为 `NULL`、`summary_revision` 为 0。每当新 memory 建立指向已有 subject 的
link，原子 add 递增该 subject 的 `new_memory_count`；link 新建 subject 的初始计数为零。本 episode 的所有结构性维护完成后，再由
1.2.4 定义的统一 summary refresh 阶段处理持久化 targets。review 修改或退役共享 memory
时，不同步刷新链接这些 memories 的其他 subjects。

Subject 审计的容量使用当前 active links 关联的 active memory 数量和 latest version
`char_count` 之和计算，不额外保存容易失去一致性的累计容量字段。

##### `subject_memory_links`


| 字段                          | 含义                           |
| --------------------------- | ---------------------------- |
| `link_id`                   | 一次 link 生命周期 ID              |
| `subject_id`                | subject                      |
| `memory_id`                 | memory unit                  |
| `link_basis`                | 建立依据：`direct` 或 `contextual` |
| `linked_at` / `unlinked_at` | link 生命周期                    |
| `opened_by_operation_id`    | 建立 link 的 operation          |
| `closed_by_operation_id`    | 结束 link 的 operation，可为空      |


`unlinked_at IS NULL` 表示 active link。同一 subject–memory 对在解除后重新建立时插入
新行，不复用旧行。`direct` 表示该 subject 是这条 memory 的组织归属：memory 是该
subject 要收的那种事实、事件、状态、决定或目标。`contextual` 表示 memory 会具体补全、
约束、更新或解释该 subject 下已归档的信息：该 subject 不是归属，但未来对该 subject
的回答需要看见这条记忆，即便两边文本不相似。每条新 memory 至少建立一个
`direct` link，且可以有多个 `direct` link；全部 active subject links 合计不得超过 5。
`contextual` link 按需建立，不表示较低优先级。
SQLite 使用 `CHECK (link_basis IN ('direct', 'contextual'))` 约束取值。

`link_basis` 在一次 link 生命周期内保持不变。需要重新分类时关闭旧 link，并以新的
`link_basis` 建立新 link。Subject review 读取当前 subject 下每条 memory 的
`link_basis`，但不修改它；两类 link 都参与 review、summary 和 search，第一版不根据
该字段过滤或调整检索分数。

任何流程新建 direct link 时，如果 LLM 判断候选 subjects 中一个是另一个的语义具体化，
只建立指向该 memory 真正归属且更具体 subject 的 direct link。该规则不禁止一条 memory
同时关联两个没有包含关系的领域，也不排除指向其他受影响 subject 的 contextual link。
包含关系只由 LLM 根据当前输入判断；程序不保存 subject 层级、不主动维护包含关系，也
不对一般性的语义包含关系作校验。Partial split 中原 subject 与本次新 subjects 之间的关系由操作类型直接确定，按下述 split 规则关闭旧 links。

Subject 分裂时：

- full split 关闭全部输入 memories 指向原 subject 的 links；partial split 只关闭被移入本次
新 subjects 的 memories 指向原 subject 的 links；
- 保留这些 memories 指向本次 split 范围之外其他 subjects 的 links；
- 对每条新建结果 link 重新判断 `link_basis`，不能直接继承原分类；partial split 中未移出的
原 links 保持不变；
- 归属在结果 subject 上的 memories 用于确定分组与命名，contextual memories 只关联到
仍存在具体联系的结果 subject；
- 每个新 subject 至少有一条 `direct` link，事务结束后每个 active memory 仍至少有
一条 active `direct` link。

memory 退役时关闭它的所有 active links，但保留 link rows 作为历史记录。

#### 1.1.7 Embedding



##### Model signature 与用途

`embedding_model_signatures` 保存 provider、model ID、revision、dimension、dtype、
normalization 和 query/document encoding mode。配置不同即视为不同 signature。

实验版 library 默认使用进程内的
`sentence-transformers/all-MiniLM-L6-v2`，固定 revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，输出 384 维 L2 归一化向量；query 与
document 都直接编码原文。模型在第一次实际编码时加载，首次使用时由 Sentence Transformers
模型仓库的 ONNX 权重和 tokenizer 下载到 Hugging Face 本地缓存，后续运行通过 ONNX Runtime
在 CPU 上复用缓存。输入按模型约束截断到 256 tokens，并使用 attention-mask mean pooling。
默认关闭 ONNX Runtime telemetry。调用方仍可显式传入其他 embedding
provider。每个 memory space 按用途选择 active signature：

- `retrieval`：memory、subject 和 query 检索；
- `boundary`：正式版的 message boundary detection。

同一次相似度计算不得混用不同 signature。模型切换时，先生成对应用途的全部必要
embedding，再原子切换 active signature。

##### `memory_embeddings`

每个 active memory 的 latest version 在 retrieval signature 下恰好有一个 embedding。
embedding 输入只包含 memory content。memory 被修改时生成新 embedding；memory 退役时
删除其检索 embedding，正文仍可用于重建。

##### `subject_embeddings`

每个 active subject 维护：

- `name`：只编码 subject name；
- `name_summary`：当 summary 非空时，编码 subject name 与当前 summary。

表以 `(subject_id, embedding_kind, model_signature_id)` 唯一标识一条 embedding，并保存
输入文本的 hash。新 subject 立即生成 `name` embedding；首次 summary refresh 才生成
`name_summary`，后续 summary 变化时更新它。公开 search 默认使用 `name`，写入阶段的
Subject 候选通道始终使用 `name`。

embedding 统一保存 dimension、source hash、vector 和 created time，并验证：

```text
byte_length(vector) = dimension * 4
```



#### 1.1.8 处理状态与 domain operation



##### `episode_extractions`

`episode_extractions` 记录 episode 是否已经完成一次逻辑上的记忆添加流程，或已经成为不应
自动重试的终态单项失败。它至少保存 episode ID、输入 hash、extractor 配置签名、状态、
完成 operation ID、完成时间，以及终态失败时的 error class 和简短原因。

`completed` 可以对应 `0..N` 条 memory，因此不能用 provenance 是否存在判断 episode
是否处理过。一个 episode 在同一 memory space 中只允许有一个 completed 结果；使用不同
配置重跑 benchmark 时创建新的 memory space。LLM 推理和 embedding 计算在事务外完成；
最终 memory、subjects、links、embeddings 和 completion record 一起提交，避免重试产生
部分结果或重复 memory。

`terminal_failure` 不产生 memory、subject、link 或 domain operation。相同来源的安全重放
返回已经记录的失败，不重复调用模型；benchmark 按来源顺序继续处理后续 session。临时传输、
限流或服务不可用在所属操作内重试耗尽后不写入该状态，而是保留未完成 episode 供显式恢复。

##### `domain_operations`

`domain_operations` 是 append-only 的结构化审计记录。operation 表示一次业务原子变化，
而不是每个单表写入。第一版至少包含：

- `add_episode_memories`
- `review_subject`
- `split_subject`
- `refresh_subject_summary`
- `retire_memory`

operation 保存 memory space、actor、规则或模型配置、简短 reason 和提交时间。
`domain_operation_effects` 统一记录受影响的 memory、memory version、subject 或 link 及其
effect type。link row 同时通过 open/close operation ID 保留关系变化来源。

`episode_summary_refresh_targets` 以 `add_episode_memories` operation 和 subject ID 为主键，
保存该 episode 的 refresh 候选及其完成 operation ID。候选包括 link 新增 memory 的 subjects、
被 review 的 subjects，以及 link/split 创建的新 subjects；full split 事务从本 episode 的集合
删除被退役的原 subject。成功 refresh 与 target 完成标记在同一事务提交，因此重放不会重复
已经完成的 refresh。

#### 1.1.9 原子事务与并发校验

以下变化分别作为一个 SQLite transaction 提交：

1. **Add episode memories**：写入 extraction completion、memory units、latest versions、
  provenance、`latest_source_at`、embeddings、subjects、links、summary refresh targets 和
   operation effects；新 subject 只写入 name 与 name embedding。
2. **Review subject**：写入全部 memory 新版本和 provenance，退役指定 memories 并关闭其
  active links，重新计算受影响 memory 的 `latest_source_at`，更新 embeddings，将当前
   subject 的新增计数置零，把被 review 的 subject 加入 refresh targets，并记录 operation
   effects；不读取或改写任何 summary。
3. **Split subject**：full split 创建结果 subjects 和 name embeddings，关闭原
  subject 的全部 links，建立结果 links 并退役原 subject；partial split 创建新 subjects、
   和 name embeddings，只关闭移出 memories 指向原 subject 的 links，建立新 links，并保留
   原 subject 的旧 summary。两种成功结果都将涉及的 active subjects 的新增计数置零，将新建
   subjects 加入 refresh targets，并记录 operation effects。
4. **Refresh subject summary**：对一个 pending target，以其全部当前 active memories 生成
   完整 summary；在成员集合、内容、link basis 和 revision 未变化时整体替换 summary、更新
   `name_summary` embedding，并把该 target 标为完成，不重置 `new_memory_count`。
5. **Switch embedding model**：新 signature 的全部必要 embeddings 准备完成后，切换对应
  用途的 active signature。

LLM 运行期间不持有数据库事务。提交前必须重新验证 memory/subject IDs、memory space、
provenance、active links、`link_basis`、输入 revision、content hash 和 model signature。
Subject review 和 split 与作用范围冲突的写入由 application 层串行执行，并使用 subject
revision 或等价 CAS 防止基于陈旧输入提交。

#### 1.1.10 Retention 与关键索引

- episode、episode blocks、memory versions、provenance、已结束 links 和 domain
operations 长期保留。
- retired memory 和 subject 保留正式内容，但不保留检索 embedding。
- 普通日志和 benchmark 中间文件不属于正式数据，可以按运行策略清理。

第一版至少建立以下查询方向的索引：

- 一个 memory space 中的 active memories 和 active subjects；
- active memories 的 `latest_source_at`；
- memory 的 latest version 和版本历史；
- episode 到 memory versions 的 provenance 反查；
- subject 到 active memories、memory 到 active subjects；
- 每对 subject–memory 最多一个 active link；
- active model signature 下的 memory 和两种 subject embeddings；
- memory space 内按时间读取 domain operations。

每个 active memory 至少一个 active `direct` link 的跨表不变量由 transaction service
校验。

#### 1.1.11 集中配置与文本容量

所有业务策略、容量和运行参数必须定义在同一份配置文件中。该文件通过公共默认值、
实验版 profile、正式版 profile 和 benchmark overrides 表达场景差异。数据格式、数据库
不变量和可从其他配置直接派生的数值不配置化。每项配置最终必须声明单位、合法范围、
是否允许为零、覆盖层级、是否进入配置签名，以及修改后是否需要重建派生数据。

LLM prompt 使用 word 数表达建议性的生成长度目标；系统不按 word 数验收，也不得在 prompt
中把该目标描述成强制要求。程序硬校验统一使用预处理后文本的 Unicode code point 数。
generation provider 的 context window 和 API 参数是外部约束，不转换为 FluxFold 的完整输入或完整
结构化输出字符上限。


| 配置或行为                         | 实验版值                                   |
| ----------------------------- | -------------------------------------- |
| subject name prompt 指导        | 建议不超过 10 words                         |
| subject name 字符硬上限            | 120 字符                                 |
| subject summary prompt 指导     | 建议不超过 200 words                        |
| LLM 新生成 subject summary 字符硬上限 | 2,000 字符                               |
| memory content prompt 指导      | 建议不超过 50 words                         |
| memory content 字符硬上限          | 1,000 字符                               |
| search query                  | 不设应用层长度上限                              |
| episode message 数量硬上限         | 256                                    |
| episode 正文总字符硬上限              | 96,000 字符                              |
| 预处理后单条 message 字符硬上限          | 默认 32,000；LongMemEval profile 为 80,000 |
| LLM 完整输入与完整结构化输出              | 不设应用层字符上限                              |


episode 字符数只计算 user/assistant message 正文，不计算 ID、时间戳或 JSON 结构开销。
实验版与 benchmark 的 dataset session 超限时明确失败，不裁剪或重新划分 session。subject
name、LLM 新生成 summary 和 memory content 超限时不得静默截断；它们构成结构化校验失败。

#### 1.1.12 LLM、embedding 与数据库运行参数

LLM 阶段参数如下：


| 阶段                | temperature | top-p |
| ----------------- | ----------- | ----- |
| memory extraction | 0.1         | 不传    |
| subject linking   | 0.0         | 不传    |
| Subject review    | 0.1         | 不传    |
| Subject split     | 0.1         | 不传    |
| Subject summary refresh | 0.1   | 不传    |


所有阶段均不设置 `max_output_tokens`。普通运行不设置 seed；generation provider 支持 seed
时，benchmark 使用 `42`。不设置全局或分阶段 LLM 并发上限，也不默认设置
requests-per-minute 或 tokens-per-minute；即默认不设置 `requests_per_minute` 或
`tokens_per_minute`。Generation provider/deployment profile 可按真实外部配额覆盖。不支持 temperature
的 generation provider profile 省略该参数。Memory Engine 不设置分阶段 request timeout、
阶段 deadline，也不在 generation provider 外再包一层 timeout。连接和单次请求 timeout 由
generation provider 的 transport 负责；timeout 归一化后在 provider 内执行下述 transport
retry，耗尽后才把错误返回 Memory Engine。

本文用 model provider 统称外部模型服务；调用 extraction、linking、review、split 和
summary refresh 的服务称为 generation provider，生成 retrieval 或 boundary vector 的服务称为 embedding provider。
SQLite 等数据库基础设施不属于 model provider。具体 provider adapter 必须先依据结构化错误
码、finish reason、响应头和 HTTP status 将厂商错误归一化，再由 FluxFold 执行统一策略；
不得只凭 HTTP status 决定错误类别。


| `error_class`                     | 典型情况                                               | 重试或修复规则                                                                  | 耗尽后的作用范围与结果                                                    |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| `transient_transport`             | DNS、连接重置、连接或单次请求 timeout 等临时传输错误                   | 按 transport retry 退避                                                     | 正式版临时暂停相关写入 pipeline；benchmark 按 1.2.6 处理                      |
| `rate_limited`                    | 临时请求或 token 速率限制                                   | 按 transport retry，优先服从 `Retry-After`                                     | 暂停到 provider 指定的恢复时间；没有有效恢复时间时使用正式版默认暂停时间                      |
| `service_unavailable`             | provider 5xx、overloaded 或临时容量不足                    | 按 transport retry 退避                                                     | 正式版临时暂停相关写入 pipeline；benchmark 按 1.2.6 处理                      |
| `authentication_or_configuration` | credential 无效、无权限、endpoint、deployment 或 model 配错   | 不重试                                                                      | 相关写入 pipeline 进入配置阻塞，等待配置变化或显式健康检查成功                           |
| `quota_exhausted`                 | 余额不足、billing 问题或硬配额耗尽                              | 不重试；provider 给出明确恢复时间时可以等待该时间                                            | 无恢复时间时进入配置阻塞；有恢复时间时临时暂停到该时间                                    |
| `invalid_request`                 | 不受支持的参数、adapter 构造了非法请求或模型能力不兼容                    | 不重试                                                                      | 相关写入 pipeline 进入配置阻塞并给出诊断；内容容量问题必须归入 `context_overflow`，不能混入本类 |
| `context_overflow`                | 完整输入超过 generation provider context window          | 不原样重试                                                                    | 当前 episode 或维护任务永久单项失败，pipeline 继续                             |
| `policy_rejected`                 | generation provider 因 safety 或 content policy 拒绝处理 | 不原样重试                                                                    | 当前 episode 或维护任务终态单项失败，pipeline 继续                             |
| `invalid_structured_output`       | JSON、schema、字段长度、非法引用或业务不变量错误                      | 带校验错误执行 structured-output 修复重试                                           | 重试耗尽后当前 episode 或维护任务终态单项失败，pipeline 继续                        |
| `incomplete_output`               | 无明确拒绝原因的空响应、截断输出或缺少必要字段                            | 按 structured-output 规则修复；若 finish reason 已表明 context 或 policy 原因，则改归相应类别 | 修复耗尽后当前 episode 或维护任务终态单项失败，pipeline 继续                        |


格式和业务均合法但事实错误或质量不理想的输出不自动重试。Extractor 明确返回“没有值得提取
的 memory”和 split 明确返回 `defer_split` 都是合法成功结果，也不重试。错误处理只重试
可恢复的调用失败和能够被确定性校验的输出错误，不把空响应、provider 错误或校验失败转换为
合法零结果。

只有 `transient_transport`、`rate_limited` 和 `service_unavailable` 执行 transport retry。
初次失败后额外重试次数为 5，初始退避为 1 秒，倍数为 2，采用 full jitter，不设置最大
退避时间。第 `retry_index` 次额外重试等待：

```text
random(0, 1 second * 2^retry_index)
```

`retry_index` 从 0 开始，provider 的有效 `Retry-After` 优先。仅由 generation provider
执行 transport retry，使用 SDK 时关闭 SDK 的重复重试。

`invalid_structured_output` 或 `incomplete_output` 在初次生成后最多额外重新生成 5 次，因此
一次逻辑输出最多生成 6 次。每次重新生成中的传输失败仍独立遵守 transport retry；不设置
跨 transport 与 structured-output 修复的 provider 调用总上限。每个结构化阶段的 system
prompt 必须给出与程序校验一致的完整判别联合字段、嵌套结构、枚举值和少量合法示例，不能用
`[...]` 代替关键契约。JSON 解析、schema、非法引用、字段长度和业务不变量错误不能伪装成
零结果或静默截断。每次修复请求由原始阶段输入、仅紧邻上一次的失败输出、精简校验反馈和
重新输出完整 JSON 对象的明确请求组成；不得嵌套此前的修复请求或累计更早输出。

发送给 LLM 的 Pydantic 校验反馈把数组下标归一为 `[*]` 并折叠重复错误，最多保留 8 类、
2,000 字符；单条说明最多 300 字符。JSON 语法错误只保留行列和解析原因，业务校验错误保留
可操作的约束说明。若 provider 没有返回响应正文，明确标记无正文，不复用更早一次输出。
Subject split 无法形成合法语义分组时使用明确的
`defer_split` 结果；声称执行 split 却违反 schema 或业务不变量的输出仍按
`invalid_structured_output` 修复。


| 配置项                                         | 值           | 含义                                              |
| ------------------------------------------- | ----------- | ----------------------------------------------- |
| `embedding_batch_size`                      | 100         | 一次 embedding provider 请求最多编码的文本数                |
| `embedding_request_timeout_seconds`         | 60 秒        | 单次远程 embedding provider 请求的 timeout             |
| `embedding_transport_max_retries`           | 5           | embedding 初次传输失败后的额外重试次数                        |
| `embedding_batch_concurrency_per_operation` | 4           | 单个 embedding 逻辑操作内部同时执行的 batch 请求上限            |
| `subject_summary_refresh_concurrency_per_episode` | 5     | 一个 episode 内同时执行的 subject summary refresh 上限          |
| `exact_scan_batch_rows`                     | 8,192       | NumPy 精确扫描时每批从 SQLite 读取并计算的 embedding 行数       |
| `sqlite_busy_timeout_ms`                    | 5,000 毫秒    | SQLite 遇到锁竞争时等待锁释放的最长时间                         |
| `sqlite_transaction_max_retries`            | 5           | 初次 SQLite transaction 冲突后的额外重试次数                |
| `sqlite_transaction_retry_initial_seconds`  | 0.01 秒      | SQLite transaction retry 的指数退避初值                |
| `sqlite_transaction_retry_multiplier`       | 2           | SQLite transaction retry 每次增长的倍数                |
| `sqlite_wal_autocheckpoint_pages`           | 1,000 pages | SQLite WAL 自动 checkpoint 的 page 阈值              |


memory 或 subject 创建、相关文本更新时立即计算对应 embedding。一个逻辑操作同时产生多个
文本时按 100 个一批合并请求，并只在该次操作内部最多并发 4 个 batch；该 semaphore 不跨
逻辑操作、memory space 或进程共享，也不是全局 provider 限流器。8 个文本形成 1 次请求，
250 个文本形成 3 次可并发请求。单个文本的批次就是 1。全部必要 embedding 在事务外生成
成功后，与正式内容和关系原子提交，不能暴露缺少当前 embedding 的 active 对象。Embedding
provider 只有归一化为 `transient_transport`、`rate_limited` 或 `service_unavailable` 的错误
才执行额外 5 次重试，并复用 generation transport retry 的退避参数；配置、权限、硬配额和
非法请求错误遵守上表的阻塞规则。

默认本地 embedding 在工作线程中执行，避免阻塞 asyncio event loop；同一 provider instance
串行调用底层模型编码。它没有网络 request，因此不应用 request timeout 或 transport retry。

SQLite 固定使用 WAL、`synchronous=NORMAL`。事务冲突最多额外重试 5 次，使用 full
jitter，不设置退避最大时间。同一 memory space 的 extraction lane 和 stateful lane 各为
单并发；stateful lane 包含候选召回、linking、正式写入以及本 episode 的全部
review/split/summary refresh。link、review 和 split 按确定性顺序串行；最终不同 subjects 的
summary refresh 可同时执行，单个 episode 上限为 5，每个成功结果仍使用独立事务提交。
本 episode 的 pending targets 未全部完成前，不释放 stateful lane。不设置跨 space 的全局
维护并发、provider 并发限制或单次 `add_episode` 的维护操作数量上限。

#### 1.1.13 Public library 与 memory-space 管理边界

实验版 public library 提供以下能力：创建或打开 memory space、删除指定 memory space、
清空全部 memory spaces、通过 `add_episode(memory_space_id, episode)` 向指定 space 添加一个
已规范化 dataset episode，以及在指定
space 中检索记忆。它还提供检索 embedding 的全量重建：先为全部 active memory latest
content、全部 active subject 的 name，以及 summary 非空 subject 的 name-summary 生成新 signature 下的向量，再在一个事务
中校验来源快照并切换 active retrieval signature。删除指定 space 和清空全部 spaces 都能通过单条管理命令完成；它们是
memory-space 级管理操作，会清除目标 space 的整套数据，不改变普通流程中 episode、历史
版本和 domain operation 的不可变约束。

Dataset adapter 只把来源格式转换为统一 episode。Memory Engine 不直接接受
LongMemEval-S 或 LoCoMo_refined 的原始 record，adapter 也不实现或复制 extraction、
Subject linking、Subject review、Subject split、正式写入或 search 逻辑。

`add_episode` 的语义是 `add_into_memory_store`，不是正式版 `add_into_buffer`。调用返回时，
该 episode 已完成全部 stateful maintenance，合法地产生零条 memory，或以异常报告一个已
持久化、可确定性重放的终态失败。同一规范化来源身份和相同 canonical payload 的重复提交是同一次逻辑 `add_episode` 的安全重放，
不重复调用模型或写入；同一来源身份对应不同 canonical payload 时拒绝覆盖原 episode。
一次 `add_episode` 不暴露部分提交的正式 memory 或 subject 状态。

Public search 的结构化结果是 library 的事实输出；供主 Agent 使用的文本由同一结构化结果
独立渲染，不得改变其中的对象、关系或顺序。同步或异步调用形式、准确函数签名、具体返回
类型和异常类由实现根据所选运行框架确定，不构成第一版领域设计约束。

### 1.2 记忆添加、关联与整理



#### 1.2.1 实验版入口与总体流程

实验版每次通过 `add_episode` 添加一个已经划定边界、不可再切分的 dataset session；LongMemEval-S 的一个
haystack session 和 LoCoMo_refined 的一个 session 分别形成一个 situational episode，
不经过 durable inbox 或 boundary detection。Episode 只包含按来源顺序排列的 user 与
assistant messages；adapter 不使用 LLM 补全、改写、猜测或按时间重新排序来源内容。

Session 时间只属于 episode。数据集没有逐消息时间时，block `observed_at` 保持空。每个
episode 持久化 `source_sequence`，明确其在 memory space 内的来源顺序；该顺序不能从时间、
source key、UUID 或数据库提交顺序推断。Dataset question、answer、evidence label 和其他
评测监督信息不得进入 episode、metadata、content hash、LLM 输入或任何会成为模型输入的
日志记录。

##### LongMemEval-S profile

- 每个 evaluation instance 建立独立 memory space。按相同数组位置组合
`haystack_session_ids`、`haystack_dates` 和 `haystack_sessions`，数组位置形成
`source_sequence`。
- Turn 的 `role` 和 `content` 直接映射为 block role 和正文；speaker 信息为空，session
date 映射为 episode `source_started_at`，逐消息时间为空。
- 不要求 session 从 user 开始、以 assistant 结束或严格交替，始终保留原数组顺序。
- `source_key` 由来源位置和 raw `haystack_session_id` 共同确定。相同 raw session ID 在同一
instance 的不同位置出现时仍是不同 episode；正文相同但时间不同的重复出现也分别保留。
- `has_answer`、`answer_session_ids`、`question`、`answer`、`question_type` 及其他评测字段
完全排除，其中 `has_answer` 不得以任何形式泄漏给 Memory Engine。



##### LoCoMo_refined profile

- 实验版读取官方公开 `conversations.jsonl` 中的结构化 `sessions`，不使用已拼接的
conversation-history 文本；每个 conversation 建立独立 memory space。
- 官方 `session_index` 是从 1 起的连续整数。`sample_id` 与该 index 共同确定
`source_key`，该 index 同时作为 `source_sequence`。
- `speaker_a` 映射为 user，`speaker_b` 映射为 assistant；保留稳定 speaker 身份和原始姓名；
`dia_id` 与 message index 作为来源定位 metadata 保存，未知 speaker 不猜测角色。
- Message text 保持为正文。非空 `blip_caption` 以明确标记的“数据集提供的图片描述”附加在
同一 message 中，即使没有图片 URL 也保留。
- 图片 URL 只作为来源 metadata，不进入正文；`query` 是搜图词而非对话或可靠视觉证据，
不保存；派生的 `has_multimodal_context` 也不保存。



##### 角色、空值、时间与 canonical identity

实验版 dataset episode 只接受 `user` 和 `assistant` 协议角色；未知角色不能被丢弃或改写成
user。多人数据同时保留协议角色与现实 speaker 身份，不能仅根据姓名或位置猜测。空白
message 不静默跳过，缺失逐消息时间和 session 时间均保持空；存在但无法解释的时间值是
无效输入，不能当作未知时间。

两套 benchmark 的非空时间均没有来源时区。实验版 profile 固定按 UTC 解释，并把该约定
写入 benchmark run manifest；这只是可复现编码约定，不声称真实对话发生在 UTC，也不使用
运行机器的本地时区。

Canonical payload 是确定性规范化后实际保存的 episode，包含 payload version、来源身份、
来源顺序、来源时间、speaker、role、正文和允许保留的 metadata；不包含内部 ID、系统时间、
评测监督字段或冗余派生字段。Canonical serialization 固定使用 UTF-8、确定的 object key
顺序、显式空值和原 message/array 顺序，再计算 SHA-256。相同输入在不同机器和运行中必须
得到相同 content hash。不同来源位置即使正文相同，也不按文本相同或相似自动去重。

当前两套 benchmark 数据已经验证不存在畸形、来源身份冲突或超过实验版容量限制的 session；
第一版 dataset adapter 不增加坏样本恢复、裁剪或容错分支。

##### `add_episode` 流程

规范化并持久化 episode 后，完整流程为：

```text
episode
→ memory extraction
→ 每条新 memory 召回 subject name，合并为批次候选池
→ 为整批新 memories 执行一个 Subject linking agent loop
→ 原子写入 memory、provenance、embedding、subject、links 和 operation
→ 先执行已触发的 Subject split / Subject review
→ 构造并持久化本 episode 的 summary refresh targets
→ 最多并发 5 个 target，分别整体重写 summary
```

LLM 与 embedding 在事务外运行。`episode_extractions` 以 episode input hash 和 extractor
配置签名保证一次逻辑完成只提交一次。Extraction 明确返回“没有有价值的 memory”时也必须
写入 completed 状态；缺失、畸形或校验失败的输出不能产生该状态。

同一 memory space 内的调用必须按 `source_sequence` 顺序提交。公共入口使用两个单并发
lane：Stage A 校验并持久化 episode、执行 memory extraction 并生成新 memory embedding；
Stage B 执行批次候选召回与 linking、原子提交和全部后续 maintenance。Stage A 完成
episode N 后立即释放 extraction lane，因此 N 在 Stage B 运行时，N+1 可以开始 Stage A；
Stage B 仍按 Stage A 的完成队列顺序进入，后序 episode 不能越过前序 episode。runner 可以
提交同一 space 的全部待处理 `add_episode`，已完成 extraction、等待 Stage B 的 prepared
results 不设容量上限。

Prepared extraction result 只存在于当前进程内存，不持久化；进程退出后根据 immutable
episode 和未完成的 `episode_extractions` 重新 extraction。Stage B 在 add transaction 前因
临时依赖或配置错误暂停时重新执行尚未提交的阶段；add 已提交后的 summary maintenance
failure 不回滚正式状态，恢复时从持久化 refresh targets 重试未完成项。暂停后该 space 不再
开始新的 extraction，其他 spaces 不受影响。
`context_overflow`、`policy_rejected`、修复耗尽的 `invalid_structured_output` 和
`incomplete_output` 由 `add_episode` 核心写入 `terminal_failure` 后仍抛出 `StageFailure`；
相同 episode 重放直接抛出同一失败而不再调用 provider，后序 episode 可以继续。

#### 1.2.2 Memory extraction

`memory extractor` 从当前 episode 提取 `0..N` 条自包含 memory unit。是否值得提取以
“遗忘是否会明显损害未来交互”为核心判断：会影响连续性、个性化、任务继续、后续指代、
状态变化或重要决定的信息应当提取；寒暄、可重新生成的通用知识、未被接受的建议、机械
操作过程和没有未来用途的重复内容通常不提取。

一条 memory 表达一个可以独立检索、更新或失效的完整对象：明确主体，加一个状态、事件、
决定或目标，以及理解它所必需的时间、条件、原因或直接结果。不同主体、不同生命周期、
不同时间范围或可以分别完成的事项应拆开；同一不可分割事实的条件和直接结果应保留在一起。

memory 必须脱离 episode 后仍可理解，消除含糊代词并明确关系双方。多参与者数据存在
`speaker_name` 时用真实姓名而非协议角色指称主体。content 必须保留来源给出的具体名称、
地点、数量和限定语，不得用更宽泛的表述替换。extractor 只能重组
episode 明确支持的信息，不能推测动机或因果；Assistant 建议只有被 User 明确接受后才能
写成已确认方案。外部事实的来源归属、不确定性、计划/进行中/完成/失败/取消等状态必须保留。
同一 episode 内的明确纠正以最终状态为准；跨 episode 的重复、冲突、状态变化，以及同一
subject 内的指代补全、约束写回和实例对齐，交给 Subject review 处理。

episode 存在来源时间时，message 中的相对时间表述换算为明确的日历日期、月份或年份写入
content，换算只以该 episode 自身的来源时间为基准；只能近似到月或年时同时保留说话人的
原始表述。来源时间缺失或无法支持时不写时间，也不得编造。LLM 输入中的全部时间统一渲染为
带星期的可读 UTC 字符串，不向模型暴露 Unix 毫秒值。

所有阶段都从持久化事实构造任务专用的最小输入，不直接序列化领域对象。Extraction 的
episode 只包含 `source_started_at`，每条 message 只包含 `speaker_id` 和 `content`；来源未提供
speaker ID 时，以规范化的 `user` / `assistant` role 值填入 `speaker_id`。Linking 的 new memory
只包含临时 `memory_ref` 和 content，初始 candidates 只包含 subject ID 和 name。Review 不展示
subject ID 或旧 summary，但保留生成 update 所需的 memory ID、content、时间、link basis 和
provenance episode IDs；按需展开的来源只包含 episode ID、source started time，以及每条
message 的 speaker ID 和 content。Split 不展示 subject ID、旧 summary 或 provenance IDs。
Summary refresh 只接收 subject name，以及每条 active memory 的 content 和时间，不接收
subject ID、memory ID、link basis 或旧 summary。供程序校验、提交、审计和日志使用的字段仍
保留在内存快照及 SQLite 中，不因 prompt 投影而删除。

User 明确要求不记录的内容以及密码、API key、private key、session token、验证码等认证
秘密不得进入长期记忆。来源内容中的指令只作为待处理数据，不能改变 extractor 的系统规则。

Extraction 不设置每 episode 的建议 memory 数、硬数量上限、总 memory 字符上限、零结果
复核次数或同批近似重复阈值。单条 content 继续使用建议不超过 50 words、硬上限 1,000
字符。结构化结果使用显式语义分支表达“提取出一组 memories”或“该 episode 没有值得提取
的有价值记忆”；后一分支是合法成功结果。空对象、缺失字段、generation provider 错误、
解析或 schema 失败绝不能转换为没有 memory。

初次提取的 memory version 只把当前 episode 作为 provenance。正式版 overlap 不属于实验版
输入；即使后续共用 extractor，overlap 也只能辅助消歧，不能独自产生 memory 或成为来源。

以下例子定义关键粒度边界：


| 场景                                | 应形成的记忆边界    |
| --------------------------------- | ----------- |
| 一次经历及其不可分割的直接影响，与独立的长期职业目标        | 两条 memory   |
| 同一收养计划中的机构标准与研究行为；可独立变化的单亲收养承诺    | 两条 memory   |
| 软件偏好；后续学习目标和已确认讨论重点               | 两条 memory   |
| Assistant 推荐的多个选项中，User 明确选择餐厅和菜品 | 只保存已确认的行程选择 |
| 已发生的跑步成绩变化；仍在进行的备赛目标              | 两条 memory   |




#### 1.2.3 新 memory 的候选召回与 Subject linking

每条新 memory 都必须 link 到至少一个 subject。Subject 表示围绕人物、项目、话题、事件
或其他可独立组织范围的一组记忆；初始名称保持宏观、简短、可独立理解，后续可以由 split
形成更具体的领域或关系 subject。

linking 首先识别每条 memory 的全部核心锚点（core anchors）。核心锚点是 memory 直接断言
或更新其事实、事件、关系、状态、决定或目标，并值得独立检索的实体或范围；一段关系可以
同时有多个核心锚点。地点、物品、属性、例子或偶然上下文中的普通提及不自动成为核心锚点，
只有 memory 对它建立了可独立使用的信息时才算。数据库中的 subject 是承接锚点 memories
的组织容器，不与核心锚点混用同一个概念。

direct link 的必要条件是目标 subject 是当前 memory 对某一核心锚点的组织归属：memory
属于该 subject 要收的那种事实、事件、状态、决定或目标，依据 subject 收的是哪类东西判断，
而不是记忆是否碰巧提到它。每个核心锚点独立执行以下顺序，一个锚点已有合适 subject 不能
替另一个锚点完成归档：

1. 只考察范围可能承接当前锚点的 candidates；属于其他锚点的合适候选不参与本锚点决策。
2. 存在合适 candidates 时，只选择 memory 真正归属的最细粒度候选；不得选择并不适合的
   更细 subject，不得为该锚点新建 subject，也不得为了重复同一归属而同时 direct 或
   contextual link 到其更粗父级。
3. 从细到粗没有任何 candidate 合适时，才以最粗可用粒度新建 subject，通常就是该实体或
   范围本身的名称，使后续 memories 汇集到同一处。

新 subject 是累积容器，不是当前 memory 的摘要；名称不得加入仅属于单次经历的日期、年份、
一次 trip、show、meeting 或 incident 等细节。只有 event 或 project 本身就是需要独立跟踪的
核心锚点时，才直接使用其名称。细粒度 subject 由多条 memories 提供稳定边界的证据后通过
split 形成，而不从单条 memory 的具体程度推导。同一批次的多条 memories 可以共同引用本次
输出中的同一个新 `subject_ref`。

例如已有 `Mike`、`Mike's Beijing trip`、`Mike's dietary preferences` 和
`John's diet habits` 时，`Mike and John are good friends` 同时以 Mike 和 John 为核心锚点：
Mike 侧 direct link 到 `Mike`；`John's diet habits` 不承接这段友情，因此 John 侧新建最粗
粒度的 `John`，不能因为 Mike 已有合适候选而遗漏 John。若已有范围同时覆盖双方的
`Mike and John's friendship`，一个 direct link 可以解决两个锚点。只有 `Melanie` 候选时，
`Melanie's family saw the Perseid meteor shower while camping in 2022` direct link 到
`Melanie`，不得新建 `Melanie's family 2022 camping trip`。

全部核心锚点决定 direct targets 后，先合并去重，再判断 contextual links。contextual link
挂到 memory 会具体补全、约束、更新或解释的既有候选 subject，包括记忆正文从未点名、只凭
常识才成立的跨域关系，例如种牙手术对饮食偏好、驾照停权对接送出行。缺失的 contextual
范围本身不触发新建，除非它同时是尚未解决的核心锚点。

被动候选召回分别以每条新 memory content 为 query，只扫描 active subject name embedding。
每条 memory 取相似度不低于阈值的前 5 个 subject，以余弦相似度作为分数放入批次对比池，
并把该 memory 的前 2 个直接放入最终池。对比池按 subject ID 去重，同一 subject 取它在所有
new memories 上的最高分，再取全局前 10 个放入最终池。最终池按“逐 memory 前 2”在先、
“全局前 10”在后去重；prompt 只展示最终池中每个 subject 的 ID 和 name，不展示分数、
summary、关联 memory 或按 memory 分组的候选视图。

本 episode 的全部新 memories 共用一个 Subject linking agent loop。模型每一轮可以返回整批
最终 `links`，也可以返回一个 `association_search(query)` 工具请求；单次 loop 最多执行 5 次
主动检索，第 6 个模型回合必须在已经累积的结果上给出最终 links。每次主动检索的候选展示
保持详细的 Subject + Memory 双通道结构：Subject 通道包含 subject name、相似度和最相关的
一条关联 memory，Memory 通道包含 memory content、相似度、时间和最相关的关联 subject；
结果按 query 分轮累积，之前轮次不会被后续结果覆盖。

主动 query 应写成某条新记忆可能改变、约束或补全的另一主题名称，而非复述 new memory。
判断依据是新记忆本身加上常识：口腔手术影响进食饮酒、驾照停权影响需要开车的行程、夜班
占用晚间。搜到的可以是被动召回漏掉的 direct home，也可以是 contextual 目标。prompt 要求
模型在输出 links 前逐条检查这种跨主题效应；不能仅因已经能选择 direct subject、正文从未
点名另一领域、或被动候选看似合理而跳过。Linking 只决定成员关系，不改写已有 memory 正文；
同一 subject 内的指代补全、约束写回、实例对齐和类别上提由 Subject review 完成。


| 配置项                                        | 值    | 含义                                             |
| ------------------------------------------ | ---- | ---------------------------------------------- |
| `subject_candidate_top_k`                  | 5    | 每条 new memory 进入初始对比池的 subject 上限              |
| `subject_candidate_direct_top_k`           | 2    | 每条 new memory 直接进入最终池的 subject 数                |
| `subject_candidate_pool_top_k`             | 10   | 按 subject 最高分进入最终池的全局上限                       |
| `subject_candidate_min_similarity`         | 0.25 | Subject 通道允许候选进入结果的最低 query-subject name 余弦相似度 |
| `association_subject_candidate_top_k`      | 8    | 每次主动检索 Subject 通道的 subject 上限                  |
| `subject_candidate_attached_memory_k`      | 1    | 每个 Subject 通道候选附带的关联 memory 数                  |
| `memory_candidate_top_k`                   | 8    | Memory 通道最多保留的 memory 数                        |
| `memory_candidate_min_similarity`          | 0.35 | Memory 通道允许候选进入结果的最低 query-memory 余弦相似度        |
| `memory_candidate_attached_subject_k`      | 1    | 每个 Memory 通道候选附带的关联 subject 数                  |
| `association_search_max_calls`             | 5    | 一个批次 linking agent loop 的主动检索调用上限              |
| `memory_link_preferred_min`                | 1    | Prompt 建议一条 memory 通常至少链接的 subject 数           |
| `memory_link_preferred_max`                | 4    | Prompt 建议一条 memory 通常最多链接的 subject 数           |
| `memory_active_subject_link_max`           | 5    | 一条 memory 可以同时拥有的 active subject link 硬上限      |


主动检索的 Subject 通道按 query 与 active subject name embedding 取前 8 个，再删除相似度
低于 0.25 的 subject。每个保留 subject 附带其 active memories 中与 query 最相似的 1 条；
没有 linked memory 时只提供 subject。

主动检索的 Memory 通道按 query 与 active memory content embedding 取前 8 条，再删除相似度低于
0.35 的 memory。每条保留 memory 附带其 active subjects 中 name embedding 与 query 最
相似的 1 个；没有 subject 时只提供 memory。

主动检索的两个通道完成后，候选按真实 subject-memory 关系组织成去重的 subject groups；
先列 Subject 通道 subjects，再列仅由 Memory 通道引入的 subjects，同一 subject 和同一组内
的 memory 只展示一次。不同新 memories 的候选视图互相独立，只在各自视图内去重，不跨
memory 去重或共享召回上限。主动结果不展示 subject summary。所有检索只使用向量相似度；
第一版不使用 BM25、全文或关键词匹配。

程序在 extraction 后为本 episode 的全部新 memories 分配临时 `memory_ref`，完成上述批次
候选池后启动一次 linking agent loop。最终输出必须覆盖每条 memory；多条 memories 可以通过
`kind = "new"` 指向同一个新 `subject_ref`。所有 links 校验成功后，程序为新 subjects 分配
正式 ID、生成 name embeddings，再把整个 episode 的 memory、provenance、subjects、links
和 extraction completion 原子提交。agent loop 的中间检索轮次不暴露正式写入或部分结果。

LLM 判断应链接哪些已有 subject、是否新建 subject，以及每条 link 的 `direct` 或
`contextual` basis。`direct` 表示该 subject 是这条 memory 的组织归属。`contextual`
是跨语义范围的检索桥：memory 的归属在别处，但它补全、约束、更新或解释该 subject 下已
归档的内容，未来 query 命中该 subject 时需要一并可见，即便两边文本不相似。

以下例子定义 direct 与 contextual 的边界。给定候选 `Mike`、`Mike's Beijing trip`、
`Mike's dietary preferences`、`John's diet habits`：


| 新 memory | 应建立的 links |
| --- | --- |
| Mike likes eating apples | `Mike's dietary preferences` direct |
| Mike bought a camera for the Beijing trip | `Mike's Beijing trip` direct |
| Mike is learning Spanish | `Mike` direct |
| Mike and John are good friends | `Mike` direct，并新建 `John` direct |
| Mike had dental implant surgery on 3 May 2024 | `Mike` direct；`Mike's dietary preferences` contextual |


种牙是医疗事件，归属是 `Mike`；记忆正文完全不提饮食，但凭口腔手术会限制咀嚼、进食和
饮酒的常识，应对 `Mike's dietary preferences` 建 `contextual`。表中把饮食 subject 列为
给定候选，只为标明正确的 basis；仅凭种牙正文做被动召回，它与饮食偏好文本不相似，通常
根本不会出现。友谊的归属是 Mike 和 John；没有 John 的 subject 时必须新建，
不能挂到 `John's diet habits`。

主动关联检索用于找出被动召回因文本不相似而漏掉、但常识表明新记忆会改变或约束的另一主题。
query 写成那个主题的名称，而不是复述新记忆：种牙检索 `Mike's dietary preferences`、
`Mike's diet plan`；驾照停权检索 `Mike's travel plans`、`Mike's commute`；医院夜班检索
`Mike's evening plans`、`Mike's sleep schedule`。搜到出行、饮食或睡眠 subject 时对其建
`contextual`。每条新 memory 至多执行一次这种搜索。

Linking 把这些记忆收进同一 subject 之后，并不改写旧正文。公开 search 按 memory content
和 subject name 排序，每个 subject 只附带 1 条 memory，且仅 Subject 通道命中的 subject
展示 summary。因此跨记忆的指代补全、约束写回、平行实例对齐和类别上提，由随后的
Subject review（可改 memory 正文并重 embed）和最终 summary refresh（只写 summary）完成。

Prompt 建议每条 memory 通常建立 1--4 个 links，硬校验要求每条 memory 至少一个
direct link，全部 active links 不超过 5。已有 subject 与其语义具体化 subject 同时
成为候选时，direct 归属只选择该 memory 真正归属且更具体的一个；该规则不排除指向其他
受影响 subject 的 contextual link。程序不校验或持久化 subject 的包含关系。

每当新 memory 建立指向已有 subject 的 active link，原子 add 递增其 `new_memory_count`；新建
subject 的初始计数为零。add 不 append memory content，也不更新 summary embedding；所有 link
目标同时进入本 episode 持久化的 refresh targets。本 episode 新建 subject 的 `summary` 为 `NULL`、`summary_revision` 为 0，
首次 summary 统一由最终 refresh 阶段生成。

#### 1.2.4 Subject review

一个 subject 自上次成功 review 或 split 后，每新增 8 条具有新 active link 的 memory，
触发一次 review：


| 配置项                                   | 值   | 含义                                                  |
| ------------------------------------- | --- | --------------------------------------------------- |
| `subject_review_new_memory_threshold` | 8   | 自上次成功 review/split 后新增到该 subject 的 memory 数量触发阈值    |
| `review_provenance_memory_max`        | 8   | 一次 Subject review provenance request 最多指定的 memory 数 |
| `memory_provenance_episode_max`       | 6   | 一个 memory version 最多关联的来源 episode 数                 |


direct/contextual 使用同一计数；同一 memory-subject 对只计一次。review 读取该 subject
当前全部 active memories，而非仅新增的 8 条。成功后计数清零，失败不清零。不按 summary
长度触发 review；本 episode 的统一 summary refresh 见本节末尾。

审核输入包含 subject name，以及每条 active memory 的 ID、content、相对当前 subject 的
link basis 和时间等必要元数据。审核允许保留、修改和全局退役 memory，不允许新建、拆分
memory、调整 links 或读写 summary。修改创建新 memory version 并重算 content embedding；退役关闭该
memory 的全部 active links，使其退出候选、review、split、summary 和公开 search，但保留
历史正文、provenance、links 与 operation。

Linking 只把记忆收进 subject。公开 search 按 memory content 与 subject name 排序，每个
subject 只附带 1 条 memory，且仅 Subject 通道命中时展示 summary。因此 review 必须编译
本 subject 内已有成员，使每条被保留的 memory 对将检索到它的 query 自洽，而不是把跨记忆
关系只写在 summary 里。适用时：

- 用兄妹记忆补全缺失的专名、地点或日期：一条写 “home country”、另一条点名 Sweden 时，
把不完整的那条改写成含 Sweden，二者仍可独立更新则不得合并。
- 把兄妹施加的约束写进被影响的那条，并带上时限：种牙限制饮酒时，改写饮酒偏好而不是把
手术改成饮食事实。
- 把将被计数或比较的平行实例改成可并列检索的句式，但不合并仍可分别完成的事项。
- 在正文中上提类别词且不发明记忆不支持的实例：Bach 与 Mozart 写成古典音乐偏好，仍保留
这两人。

相对时间需要对齐、或仅凭 content 无法判断冲突是否真实时，先请求 provenance。

审核进行一至两次结构化输出。`provenance_request` 与最终 `review` 是两个显式合法的
结构化分支。首次请求的 `requested_provenance` 为 null；仅在 content 和元数据不足以解决
重复、冲突、纠正、状态变化、信息归属或若干
memory 必须共享的日期，且来源会改变判断时，首轮可以返回完整的 `provenance_request`
结果。一次最多请求 8 个不同
memory IDs，系统返回这些 memories 当前 provenance 涉及的全部 episodes，不设置
`review_provenance_episode_max`。第二次请求的 `requested_provenance` 为 non-null，此时只能
输出最终 `review`，不能再次请求 provenance。

最终结果只列出 memory updates 和 memory retirements。未列出的 memory 保持不变。update
可以替换 content、provenance 或两者，但必须至少改变一项；提供
provenance 时，它表示包含 1--6 个不同有效 episode IDs 的完整替换集合。超过 6 不能截断；
无法由至多 6 个来源准确支持的合并不得执行。

只允许拼接共同描述同一个、不可独立更新事实的 memories。不同但相关、可以分别变化的事实
继续分开，包括日后同一问题的两跳。审核不能默认新事实覆盖旧事实，也不能仅因时间较早
退役；无法解决的冲突应保留，由最终 summary refresh 准确表达不确定性。共享 memory 的内容
或生命周期修改立即全局生效，但不把链接这些 memories 的其他 subjects 加入 refresh targets。

完成本 episode 触发的全部 split 与 review 后，系统统一刷新以下 subjects：link 导致 memory
新增的 subjects、被 review 的 subjects、以及 link 或 split 创建的新 subjects；full split
退役的 subjects 从集合中排除。这个集合不包含仅因 review 更新或退役共享 memory 而受影响的
其他 linked subjects。targets 归属于本 episode 的 add operation，并在结构性事务中持久化；
同一 subject 只保留一个 target。

每个 target 单独调用一次 LLM，输入只包含 subject name 和全部当前 active memories 的
content 与时间，明确不提供 subject ID、memory ID、link basis 或旧 summary；输出是基于这些 memories 的完整新
summary。提交时重新校验成员集合、memory content、link basis 和 `summary_revision`，随后
整体替换 summary、更新 `name_summary` embedding，并原子标记 target 完成。refresh 不清零
`new_memory_count`，因此不会延后后续 review。

同一 episode 的不同 targets 最多并发 5 个；某个 refresh 失败时，其他已经开始的 refresh
继续完成，成功项各自提交并保持完成状态。任何未完成项都使本 episode 形成可恢复的
maintenance failure，不回滚已经提交的 add、review 或 split，并暂停当前 space 的后续处理。
安全重放从持久化 targets 重建集合，只重试未完成且仍 active 的 subjects，不重复已完成项，
也不重跑 extraction 或 linking。

#### 1.2.5 Subject split

split 只在新 memory 建立指向某 subject 的 active link 时检查。共享 memory 因其他 subject
review 而更新或退役、或没有新增 link 的其他容量变化，不触发检查。


| 配置项                                          | 值         | 含义                                                                  |
| -------------------------------------------- | --------- | ------------------------------------------------------------------- |
| `subject_split_memory_count_threshold`       | 24       | subject 的 active memory 数量触发 split 的阈值                              |
| `subject_split_total_memory_chars_threshold` | 8,000 字符 | subject 下 active memory latest content 总字符数触发 split 的阈值             |
| `subject_split_result_subject_min`           | 2         | full split 的新 subjects 数或 partial split 的原 subject 加新 subjects 总数下限 |
| `subject_split_result_subject_max`           | 5         | full split 的新 subjects 数或 partial split 的原 subject 加新 subjects 总数上限 |
| `subject_split_result_min_memories`          | 3         | 每个新建结果 subject 至少必须包含的 memory 数                                     |
| `subject_split_result_target_memory_max`     | 20        | 每个新建结果 subject 的 memory 数硬上限                                          |
| `subject_split_memory_membership_max`        | 2         | 一条输入 memory 最多可以归属的本次新建 subject 数                                   |


active memory 数达到 24，或 latest contents 总字符数达到 8,000，即满足触发条件；字符
条件不要求另一个最小 memory 数。两个数值都是触发整理的软阈值；实验版不限制单个 subject
最终关联的 memory 数量或总字符数，不设置硬容量、动态阈值、冷却期或封存状态。每次又有
memory link 到已达到任一阈值的 subject 时，都重新尝试 split。

split 的结构化结果只能是 `full_split`、`partial_split` 或 `defer_split`，按以下排他顺序选择；
仅在结构上能凑出分组不代表该分组具有组织意义：

- **full split** 仅用于全部输入 memories 都能自然归入 2--5 个有意义、可独立检索、更新和
增长的更细 subjects，且没有仍需原粗粒度 subject 承接的残余 memory。原 subject 的每条
active memory 至少进入一个、最多进入两个新 subjects；原 subject 被完整替换并退役。不得为
覆盖完整而把离群 memory 强塞进某组或创建 catch-all。
- 无法 full split 时，**partial split** 仅用于 1--4 个有意义、可独立增长的群组已经突出，
但其余 memories 没有共同的更细范围、仍需原粗粒度 subject 承接。它创建新 subjects，同时
保留原 subject 的 ID、name 和 active 状态。LLM 只列出新 subjects 的 name，以及应移入
它们的 memory IDs 和 link basis。程序取
所有被列出 memory IDs 的并集，关闭它们指向原 subject 的 links；未被列出的 memories
继续留在原 subject。移出集合必须非空且不是原集合，原 subject 至少保留一条 memory；
不得为扩大新群组而把残余 memories 强行移出。
- 以上两者都不适用时使用 **defer split**，包括没有连 3 条 memories 都能组成的连贯群组、
有意义的分组会违反任一结果约束，或表面群组不是值得独立检索、更新和增长的稳定范围。它是合法
业务结果，不修改 subjects、memories、links 或计数，并记录 warning；下次又有 memory
link 到原 subject 且容量仍达到阈值时再次尝试。它计入 Subject split 触发次数，不计入
benchmark 失败样本数。

每个新 subject 必须包含 3--20 条不同 memories，并且至少有一条 `direct` link；上下限均由
程序硬校验。
被移出原 subject 的每条 memory，在关闭指向原 subject 的 link 之后，仍必须至少有一条 active
`direct` link：来自本次新 subjects 的新 direct，或仍指向 split 范围之外其他 subjects 的既有
direct。一条 memory 在本次新 subjects 中最多出现两次；partial split 中出现在任一新 subject 的
memory 到原 subject 的 link 由程序关闭，但它指向 split 范围之外其他 subjects 的 links
保持不变。不设置结果 subject 的字符目标、总 link 倍数或整体重叠率上限。

分组依据是未来是否需要独立检索、更新和增长，而不是平均分配数量。日后同一问题需要同路
检索的记忆应留在一起，例如搬家与点名来源国的事实、将被计数的平行实例；若必须拆开，保
留仍成立的 contextual link。结果 name 应保留原
主体锚点并表达具体领域、项目模块、事件阶段或人物关系；不得使用没有语义边界的 Other、
Misc 等名称。结果 subjects 存在语义包含关系时，LLM 只把 memory 分配给它真正归属且更
具体的一个；程序不校验或维护这种包含关系。所有新 links 的 `link_basis` 重新判断，不能
继承原值；contextual memory 只进入仍有具体联系的新 subject。

例如，原 subject `Mike` 中只有一组 memories 足以形成 `Mike's dietary preferences` 时，
partial split 把这组 memories 移入新 subject，其余无共同具体领域的 memories 继续留在
`Mike`。被移走的 memory 不再同时 link 到 `Mike`，但可以继续 link 到与本次 split 无关的
其他 subjects。

full split 和 partial split 都在一个事务中提交新 subjects、name embeddings、refresh targets
和 link 变化，memory content 与 provenance 不变；不读取或生成 summary。成功后所有结果
active subjects 的新增计数归零；partial split 保留原 subject 的旧 summary，直到最终 refresh。
Review 与 split 同时满足时先执行 split：
full 或 partial split 成功后无需立即 review；defer split 后仍执行已经达到触发条件的 review。

JSON、schema、字段类型或非法 ID 错误归入 `invalid_structured_output`，按 1.1.12 的规则
立即反馈并修复。结果虽然符合 schema 但违反 ID 覆盖、结果数量、最少 memories、新 subject
缺少 direct、移出后某条 memory 不再有任何 active direct，或 membership 等业务不变量时也
归入同类；不能静默改写为 defer split。LLM 判断不存在有意义的合法分组时应直接输出
`defer_split`。

#### 1.2.6 Benchmark 执行与可复现性


| 配置项                                            | 值       | 含义                                                                |
| ---------------------------------------------- | ------- | ----------------------------------------------------------------- |
| `benchmark_memory_space_build_concurrency`     | 10      | benchmark 同时构建的独立 memory space 数                                  |
| `benchmark_search_concurrency`                 | 5       | 同时执行的 benchmark search/QA 样例数                                     |
| `benchmark_seed`                               | 42      | generation provider 支持 seed 时 benchmark 使用的固定 seed                |
| `benchmark_memory_space_build_timeout_seconds` | 7,200 秒 | 构建一个 memory space 的总 timeout                                      |
| `benchmark_search_sample_timeout_seconds`      | 300 秒   | 一条 benchmark search/QA 样例拿到并发槽之后，search + 生成的 timeout；不含排队等待 |
| `benchmark_checkpoint_interval_items`          | 1       | 每完成多少个项目保存一次 checkpoint 和结果                                       |


LoCoMo 的 10 个 conversations 分别建立 10 个 memory spaces，可同时构建。LongMemEval 的
500 个 evaluation instances 分别建立独立 memory space；不同 instance 的 haystack 不能
合入同一 space。

Benchmark runner 按 session `source_sequence` 创建公共 `add_episode` 调用，不调用
`_prepare_add`、`_commit_prepared_add` 或其他私有阶段 API。同一 space 的 extraction 单并发；
前序 session 完成 extraction 并进入 stateful lane 后，后序 session 可以开始 extraction。
候选召回、linking、正式写入、review、split 和 summary refresh 在同一 stateful lane 中按
来源顺序完成；单个 episode 的 summary targets 可以按上限 5 并发，后序结果不能越过仍在处理的前序 session。prepared results、等待调用和已
完成但待 stateful 处理的数量不设上限。前序 session 成为终态单项失败后，由共享
`add_episode` 核心记录失败并继续推进来源顺序。不同 spaces 的两条 lane 可以并行。

不设置整次 benchmark 总 timeout。每完成一个 episode ingestion、space ingestion 或一条
QA 都原子保存 checkpoint；prepared extraction 不写 checkpoint，崩溃恢复后重新 extraction。
answer 以 `predictions.jsonl` 中已写入的 question ID 为 checkpoint：每完成一题立即追加
prediction 与对应 `search_results.jsonl` 记录；再次运行同一 run 的 answer 时跳过这些 ID，
不删除已有预测。`benchmark_search_sample_timeout_seconds` 从该题拿到
`benchmark_search_concurrency` 槽之后起算，只覆盖这一题的 search 与生成，排队等待不计入。
Benchmark runner 不提供完整 item 外层 retry；generation transport、
embedding transport、structured-output repair 和 SQLite transaction 只执行各自所属操作内
的有限重试。模型给出的结构和业务均有效但错误的答案不重试。一次 full 或 sample 脚本只
执行一个 run。build 未指定 `--run-dir` 时，在 `runs/` 下创建
`{dataset}_{月}.{日}_{HH:MM}_{seq}` 目录：时刻为本地墙钟时间，月日不补零、时分补零，
同一分钟内多次 build 递增 seq，例如 `runs/longmemeval_8.27_21:02_1` 与
`runs/longmemeval_8.27_21:02_2`。answer 与 score 未指定 `--run-dir` 时，使用同一
dataset、同一 mode（sample 或 full）下按该命名解析出的最新目录；目录名无法解析、缺少
manifest，或 manifest 的 dataset/mode 不匹配的项不参与选择。未指定 `--run-dir` 时必须
提供 `--dataset`。需要指向特定 run，或恢复未完成的 build / answer 时，显式传入 `--run-dir`。
runner 不内置重复次数，也不跨 run 计算均值或标准差。

每套数据集提供 `build`、`answer`、`score` 三个独立 stage，并分别提供 full 与 sample
薄脚本，共六个可直接通过 `python -m benchmarks.scripts.<stage>_<mode>` 运行的模块。
build 只构建数据库和写入审计产物；answer 从同一 run manifest 和数据库执行 public
search 并生成官方字段形状的 predictions；score 独立读取 predictions 生成逐题结果和汇总。
build 把 dataset hash、选择范围、配置签名、embedding model 和 seed 写入不可变
manifest，供复盘。answer 和 score 只校验 dataset 与 space 选择与 manifest 一致，不要求
当前 FluxFoldConfig 与 build 时相同。manifest 记录 build 使用的 generation model，供复盘
写入侧；answer 和 score 各自读取独立的 generation provider 配置，不要求与 build model
相同。
数据集由 `./scripts/setup-dev.sh` clone 到 `data/`：
`LoCoMo_refined` 与 `LongMemEval` 来自其上游 Git 仓库，LongMemEval-S 的
`longmemeval_s_cleaned.json` 另从 Hugging Face 下载。build 默认读取这些本地文件，可用
`--data-path`、`--conversations-path`、`--questions-path` 覆盖；build/answer/score 运行时
不下载数据集。Benchmark 脚本从仓库根目录加载 `.env`；进程里已经存在的环境变量优先。

LoCoMo_refined sample 必须选择一个 conversation ID 或零基位置，并处理该 conversation 的
全部 sessions 和全部 questions。LongMemEval-S sample 可以显式选择 question IDs；未指定时
确定性选择覆盖 `abstention`、`knowledge-update`、`multi-session`、
`single-session-assistant`、`single-session-preference`、`single-session-user` 和
`temporal-reasoning` 的七个完整 evaluation instances。full mode 始终处理全部数据。

build、answer 和 score 分别从 `.env` 读取一组 generation provider 配置
（`FLUXFOLD_BUILD_*`、`FLUXFOLD_ANSWER_*`、`FLUXFOLD_SCORE_*`，每组包含 `MODEL`、
`API_KEY`、`BASE_URL`）。retrieval embedding 默认使用上述本地模型；设置
`FLUXFOLD_EMBEDDING_PROVIDER=openai-compatible` 时，改为读取独立的 embedding model、
dimension、revision、API key、base URL 和 query/document encoding mode 配置。
LongMemEval 输出 `question_id`/`hypothesis`，使用官方 `evaluate_qa.py` 的 yes/no LLM judge prompt。
LoCoMo_refined 输出 `qa_id`/`predicted_answer`，使用官方 `refined` LLM judge prompt、token F1 和 BLEU-1；多个合法 reference 取最佳
匹配。answer 阶段按数据集选择 prompt：LoCoMo_refined 要求短短语、尽量使用记忆原文、保持时间粒度并把相对时间锚定到记忆日期；LongMemEval 要求覆盖全部所需事实，并在有 `question_date` 时写入 `Current Date`。检索结果渲染为 subject 分组，每条 memory 带上 `latest_source_at` 对应的日期（`D Month YYYY`）。

上述临时错误在所属操作内重试耗尽后暂停当前 memory space 的 benchmark 记忆构建流水线；
`rate_limited` 或
`quota_exhausted` 给出明确恢复时间时暂停到该时间，否则进入临时暂停并等待下一次显式恢复
探测，恢复后从最早未完成项目继续。`authentication_or_configuration`、没有恢复时间的
`quota_exhausted`、`invalid_request` 或不可恢复的数据库配置/存储错误使 benchmark 进入
配置阻塞，不能用定时 retry 代替人工修复。`invalid_structured_output`、
`incomplete_output`、`context_overflow` 或 `policy_rejected` 只使对应 episode、维护任务或
QA item 成为终态失败，不暂停其他 spaces，也不永久阻塞当前 space 的后续来源顺序。

Add 的 memory、links 和 episode completion 一旦原子提交即永久成功，后续 maintenance
失败不回滚它们。显式恢复 build 时，同一 episode 的幂等重放跳过 extraction 与 linking；
已经提交的 split/review 由当前 subject 状态和计数避免重复执行，随后从
`episode_summary_refresh_targets` 读取仍 active 且未完成的 targets。并发 refresh 中已经成功
提交的 targets 带有完成 operation ID，恢复时不会重复执行；失败或尚未开始的 targets 会被
重新构造并重试。benchmark checkpoint 只有在 episode 的全部 maintenance 完成后才跳过该
source sequence。

第一版先完整实现 extraction、Subject linking、Subject review、Subject split、Subject summary
refresh 和 search，
再执行正式实验；ablation study 留到以后。实验按配置直接报告各项结果，不选择优胜配置，
也不要求把模型因素与系统设计因素隔离。

每个 benchmark question 使用原始问题文本直接执行一次 public search，不调用 LLM 改写
query，也不执行迭代检索。除数据集官方 QA 指标外，实验只额外报告：

- 记忆写入与整理过程分别报告 `successful_llm_call_count` 和
`failed_llm_call_count`；只有通过 provider 调用及当前结构化校验的结果计为 success，provider
错误或需要 structured-output repair 的响应计为 failed；
- 上述成功和失败调用中 provider 已报告 usage 的 token 量，分别报告
`build_llm_input_tokens`、`build_llm_output_tokens` 和 `build_llm_total_tokens`。
`input_tokens` 对应 provider 的 prompt/input tokens，`output_tokens` 对应
completion/output tokens，`total_tokens` 使用 provider 报告的总量，不由前两项相加。
benchmark QA 回答和评测调用不计入该写入整理总量；
- public search 检索延迟；
- active memory 数和 active subject 数；
- Subject split 触发次数和 Subject review 触发次数；
- 失败样本数。

每个 memory space 完成全部 session 处理并最终定型后统计记忆压缩率：

```text
source_chars = 实际交给 extractor 的全部规范化 episode 正文字符数
memory_chars = 全部 active memory latest contents 字符数
             + 全部 active subject summaries 字符数
memory_compression_rate = 1 - memory_chars / source_chars
```

`source_chars` 包含作为正文输入的图片描述，不包含 role、时间、metadata 或评测字段。
`memory_chars` 不包含 retired memories、历史 memory versions 或 retired subjects；这些对象
只用于审计和追溯。实验逐 memory space 报告 `source_chars`、`memory_chars` 和压缩率。

`build_summary.json` 还按 memory space 报告当前组织快照：`episode_count`、`active_subjects`、
`direct_links`、`contextual_links`、`retired_memories`、`retired_subjects`、`rewritten_memories`
（latest version 号大于 1 的 active memories）、`max_links_per_memory`、
`max_direct_links_per_memory`、`max_contextual_links_per_memory`、`mean_links_per_memory`、
`max_memories_per_subject`、`mean_memories_per_subject`、`max_provenance_per_memory`，以及
`subjects` 列表。列表按 name 再按 subject ID 排序，每项只含 `name`、`direct_links` 和
`contextual_links`。link 与 provenance 计数只包含 active memories 指向 active subjects 的
active links，以及 latest memory version 的 provenance。没有 active memories 或 active
subjects 时，对应 max/mean 为 0。

#### 1.2.7 实验版记忆构建日志

每次 memory-space build 生成四份日志：一份 JSONL 结构化事件日志，用于机器分析、统计和定位失败；一份 Markdown 高可读性审计日志，用于人工完整复盘 memories 和 subjects 如何形成及演化；一份 Markdown LLM 输入输出采样日志，用于人工阅读完整 prompt 与模型输出；一份覆盖写入的 Markdown 当前记忆库快照，按 memory space 组织展示当前全部 active subjects 的 name 与 summary，以及各自 active linked memories 的 content。结构化事件日志与审计日志共享 build/run ID、memory-space ID、episode ID、source sequence、domain operation ID 及正式对象 ID；LLM 采样日志通过 `run_id` 与 `request_id` 对应到同一次 `llm_call`。记忆库快照不进入结构化事件 JSONL。

这四份日志是实验产物，不是 SQLite 正式数据或可写事实源，不能反向驱动记忆状态，也不能
进入后续 LLM 输入。高可读性日志和记忆库快照包含完整 memory 内容；审计日志还包含完整
benchmark 对话以及完整 LLM prompt 与输出，必须按包含原始对话数据的敏感实验产物保存。

##### 结构化事件日志

结构化日志记录关键事件，不承担完整正文快照。每条事件至少包含时间、severity、事件类型、
关联 IDs、状态或结果，以及适用的对象数量、耗时和简短原因。覆盖范围至少包括：

- memory-space build 的开始、完成、暂停、恢复和失败；
- episode 开始处理、extraction 完成、Subject linking 完成和原子提交；
- 一次 episode 提取、创建和写入的 memory 数，新建 subject 数及新建 link 数；
- 本 episode 的 association search 详细结果合计涉及的唯一 candidate memory 数；初始
name-only candidates 不包含 memory，因此未调用主动检索时该值为 0。若至少一次调用
association search，还记录 `association_search_called = true`，以及主动结果相对初始候选额外
引入的唯一 candidate memory 数；
- 记忆写入与整理阶段每次 LLM 调用的阶段、attempt、success/failed 结果、耗时，以及
`input_tokens`、`output_tokens` 和 `total_tokens`；不记录 generation provider
使用的其他 token 子类别；
- Subject review 和 Subject split 的触发、开始、完成或失败，以及涉及的 subject ID；
- Subject review 完成事件记录该次 review 是否请求并查看了 provenance；
- subject summary refresh target 的建立、开始、完成或失败，以及涉及的 subject ID；
- split 的 `full_split`、`partial_split` 或 `defer_split` 结果；
- review 导致的 memory 更新数和全局退役数；
- model provider `error_class`、structured-output retry、warning、终态单项失败、临时暂停、
配置阻塞和恢复；
- partial split 移出的 memory 数、新建 subject 数，以及 full split 退役的原 subject。

结构化日志保留正式 IDs 和计数，便于汇总实验指标；不为了日志给 memory 增加 name 字段。

##### 高可读性审计日志

高可读性日志按 episode 来源顺序和后续维护实际发生顺序展开。它必须展示已经通过校验并
参与正式状态决策的完整内容，而不只是计数或对象 ID。

每个 episode 在 extraction 和 Subject linking 完成后展示：

- extractor 实际读取的规范化 episode 内容，包括 message 顺序、speaker、role、已知来源
时间和正文；不包含 dataset question、answer、evidence 等评测监督字段；
- extraction 得到的全部 memory contents，以及每条 memory 最终 link 到的 subject names；
如果结果是 `no_valuable_memory`，明确展示该语义结果；
- 本 episode linking 新建 subjects 的 name；
- 本批原子提交的最终结果。

memory unit 没有正式 name。日志为同一段落内的 memories 分配 `M1`、`M2` 等仅供阅读的
短标签，并同时展示完整 content；短标签不能保存为领域字段或跨操作身份，跨段落引用使用
正式 memory ID。

每次 Subject review 展示：

- review 前的 subject name，以及全部 active memories 和 link basis；
- provenance request 和系统返回的来源 episodes（如果发生）；
- 每条被修改 memory 的正式 ID、修改前 content、修改后 content，以及 provenance 的前后
完整集合；
- 每条全局退役 memory 的 content，以及因此关闭的全部 active subject links；
- 未改变的 memories 可以按 ID 和 content 列出一次，无需伪造 change；
- review 后的最终 active memory 集合。

每次 Subject split 展示：

- split 前原 subject 的 name，以及全部 active memories 和 link basis；
- `full_split`、`partial_split` 或 `defer_split` 的结果和理由；
- full split 后全部新 subjects 的 name、memory membership 和 link basis；
- partial split 新建的 subjects、移出的 memories，以及继续留在原 subject 的 memories；
- defer split 的 warning 和后续仍会在新增 link 后重试的说明。

每次 Subject summary refresh 展示 subject name、参与重写的全部当前 active memories 和
link basis，以及生成的完整新 summary；审计输入不展示旧 summary。

高可读性日志还按实际发生位置展示 model provider 错误类别、structured-output 修复重试、
warning、error、终态单项失败、临时暂停、配置阻塞和恢复，使一次 memory-space build 可以
仅凭该日志按时间顺序复盘。

##### LLM 输入输出采样日志

该日志单独写入 `llm_io_samples.md`，不进入结构化事件 JSONL。它记录已经通过结构化校验的
**首次 attempt** 成功调用：完整 system prompt、完整 user prompt（原始阶段输入，不含
repair 包装）和完整模型输出。需要 structured-output 修复才成功的调用不采样。

按出现顺序采样，满额即停：

- `extract`、`link`、`review`、`split`、`summary` 各 2 次。`link` 只采未调用
`association_search` 的单轮路径；`review` 只采未请求 provenance 的单轮路径。
- 若发生 `association_search`，额外采样 1 个完整 agent loop：记录每次 linking prompt、对应
的 `association_search` 请求，以及累积全部 association 候选后的最终 linking 结果；轮数为
实际工具调用次数加一，最多 6 轮。
- 若发生 `provenance_viewed`，额外采样 1 次完整两轮：第 1 轮 review prompt 与
provenance 请求，第 2 轮带上来源 episodes 后的 review prompt 与最终 review 结果。

某类调用在本次 build 中未出现则该项空缺。显式恢复续跑时，已写入文件的样本计入配额，
不因引擎重启而重复超过上限。JSON 形态的 prompt 与输出按缩进展开，便于阅读。

##### 当前记忆库快照

每次 episode 的 extraction、linking 与本 episode 维护全部成功完成后，覆盖写入
`memory_bank.md`。整次 memory-space build 成功结束时再写一次最终快照。该文件只展示当前
正式状态：按 `space_key` 分组的全部 active subjects，每个 subject 给出 name 和 summary，
其下列出当前 active linked memories 的 content。不包含 subject ID、memory ID、link basis、
retired 对象、来源 episode 或过程事件。同一 memory 若同时链到多个 subjects，在每个
subject 下各出现一次。新建尚未 refresh 的 subject 可以没有 summary 正文。该快照反映写完
当下的 SQLite 状态，不是按时间追加的过程日志。

### 1.3 记忆检索



#### 1.3.1 公开 `search`

`search(query)` 只读取 active memories、active subjects 和 active links。第一版所有通道
仅执行精确向量检索，不使用 BM25、关键词、融合、重排或第二轮筛选。query 不设应用层字符
上限，也不静默裁剪。


| 配置项                                    | 值    | 含义                                                |
| -------------------------------------- | ---- | ------------------------------------------------- |
| `search_subject_top_k`                 | 5    | 公开 search 的 Subject 通道最多召回的 subject 数             |
| `search_subject_min_similarity`        | 0.25 | 公开 search 的 Subject 通道最低 query-subject name 余弦相似度 |
| `search_subject_attached_memory_k`     | 1    | 每个公开 search Subject 通道结果附带的关联 memory 数            |
| `search_memory_top_k`                  | 15   | 公开 search 的 Memory 通道最多召回的 memory 数               |
| `search_memory_min_similarity`         | 0.35 | 公开 search 的 Memory 通道最低 query-memory 余弦相似度        |
| `search_memory_attached_subject_k`     | 1    | 每个公开 search Memory 通道结果附带的关联 subject 数            |


Subject 通道使用 subject name embedding，召回最多 5 个相似度不低于 0.25 的 subjects；
每个 subject 附带 active linked memories 中与 query 最相似的 1 条。Memory 通道召回最多
15 条相似度不低于 0.35 的 active memories；每条 memory 附带 active linked subjects 中
name embedding 与 query 最相似的 1 个。

通过各通道 top-k 和阈值的所有对象都进入最终结果。合并后按真实 links 组织为去重的
subject groups：先保持 Subject 通道排名，再按 Memory 通道首次引入顺序追加其他 subjects；
每个 subject 只展示一次，组内同一 memory 也只展示一次。只有 Subject 通道直接命中的最多
5 个 subjects 携带 summary；仅由 Memory 通道引入的 subjects 只携带 name。结构化结果同样
不暴露这些额外 subjects 的 summary，不只是文本 render 隐藏。不计算融合分数、不重新排序、
不再次淘汰。direct/contextual links 均参与，第一版不调整权重。由通道数量可派生出去重前
最多 20 个 subjects 和 20 条 memories，不把该结果重复配置为另一个上限。

不设置最终 subject 数、memory 数、每 subject memory 数、返回文本总字符数、单条 summary
字符数、单条 memory 返回字符数或超限最小保留数量。Public library 返回结构化结果；面向
主 Agent 的文本仅是同一 subject-group 结构的确定性呈现。

#### 1.3.2 NumPy 精确扫描

实验版不使用 query embedding cache。检索从 SQLite 分批读取当前 memory space、active
retrieval signature 和目标实体下的全部向量，每批最多 8,192 行。归一化 query 向量与
目标向量矩阵使用 NumPy 矩阵乘法计算 cosine similarity，并持续维护全局 top-k；分批只
限制内存，不改变所有向量参与计算的精确语义。相同浮点值以稳定 ID 决定顺序，不设置浮点
近似并列容差。

### 1.4 LLM 结构化输出参考草图

本节记录五个记忆构建阶段当前使用的精确结构化输出契约。它们不是 public API，但字段名、
判别联合编码和枚举值必须与程序校验及 system prompt 一致；修改契约时三处同步修改。Prompt
可以补充模型相关说明，但不得省略本节的完整结构和合法示例。

#### 1.4.1 Memory extraction

Extraction 使用判别结果明确区分提取成功与“没有有价值的记忆”。有结果时 `memories`
至少包含一项：

```json
{
  "result": "memories",
  "memories": [
    {"content": "A self-contained memory supported by this episode."}
  ]
}
```

没有值得提取的内容时使用独立语义分支：

```json
{
  "result": "no_valuable_memory",
  "reason": "The episode contains no information whose loss would harm future interaction."
}
```

程序在成功 extraction 后为每条 memory 分配批次内临时 `memory_ref`；该引用不是 extractor
生成的正式 memory ID。畸形对象、缺失判别字段或 `result = "memories"` 但数组为空均不是
`no_valuable_memory`。

#### 1.4.2 Batched Subject linking

一个最终 linking 结果覆盖本 episode 的全部 new memories。新 subject 使用本次输出内稳定的
`subject_ref`，多条 memory 可以通过 `new` 目标共同引用它：

```json
{
  "result": "links",
  "new_subjects": [
    {
      "subject_ref": "new_subject_1",
      "name": "Mike's dietary preferences"
    }
  ],
  "links": [
    {
      "memory_ref": "memory_1",
      "subject": {"kind": "existing", "subject_id": "subject-uuid"},
      "basis": "direct"
    },
    {
      "memory_ref": "memory_1",
      "subject": {"kind": "new", "subject_ref": "new_subject_1"},
      "basis": "direct"
    },
    {
      "memory_ref": "memory_2",
      "subject": {"kind": "new", "subject_ref": "new_subject_1"},
      "basis": "direct"
    }
  ]
}
```

如果 agent loop 使用主动关联检索，每轮中间请求采用以下形状；一个 loop 最多返回 5 次该
形状，随后仍需返回覆盖整批 memories 的最终 linking 结果：

```json
{"result": "association_search", "query": "the other topic this memory could change"}
```

程序校验所有临时引用、已有 IDs、每条 memory 的 direct link 和 link 数量约束，并在应用前
验证 existing IDs 属于初始 `candidates` 或任一累积的 `association_search_results`。初始
candidates 只包含 subject ID 和 name；主动结果保留详细的 Subject/Memory 双通道候选，但
仍不包含 subject summary。

#### 1.4.3 Subject review

首轮只有在现有 content 与 metadata 不足以判断时才请求 provenance：

```json
{
  "result": "provenance_request",
  "memory_ids": ["memory-uuid-1", "memory-uuid-2"]
}
```

无需 provenance，或系统返回请求的 episodes 后，模型输出最终 review。每个 change 使用
明确的 `keep`/`replace` 动作，避免通过字段缺失猜测含义；列入 `updates` 的对象必须实际
改变 content、provenance 或两者之一：

```json
{
  "result": "review",
  "updates": [
    {
      "memory_id": "memory-uuid-1",
      "content_change": {
        "action": "replace",
        "content": "Updated self-contained memory."
      },
      "provenance_change": {"action": "keep"}
    },
    {
      "memory_id": "memory-uuid-2",
      "content_change": {"action": "keep"},
      "provenance_change": {
        "action": "replace",
        "episode_ids": ["episode-uuid-1", "episode-uuid-2"]
      }
    }
  ],
  "retirements": ["memory-uuid-3"]
}
```

取得 provenance 后不能再次请求。未列入 `updates` 或 `retirements` 的 memory 保持不变；
retirement 的全局语义和 provenance 完整替换语义以 1.2.4 为准。

#### 1.4.4 Subject split

Full split 输出全部新 subjects 及其完整成员关系：

```json
{
  "result": "full_split",
  "subjects": [
    {
      "subject_ref": "new_subject_1",
      "name": "Mike's dietary preferences",
      "links": [
        {"memory_id": "memory-uuid-1", "basis": "direct"},
        {"memory_id": "memory-uuid-2", "basis": "direct"},
        {"memory_id": "memory-uuid-3", "basis": "direct"}
      ]
    },
    {
      "subject_ref": "new_subject_2",
      "name": "Mike's travel plans",
      "links": [
        {"memory_id": "memory-uuid-4", "basis": "contextual"},
        {"memory_id": "memory-uuid-5", "basis": "direct"},
        {"memory_id": "memory-uuid-6", "basis": "direct"}
      ]
    }
  ]
}
```

Partial split 只列出新 subjects 和要移走的 memories；原 subject 的最终成员集合由程序作
集合差得到：

```json
{
  "result": "partial_split",
  "new_subjects": [
    {
      "subject_ref": "new_subject_1",
      "name": "Mike's dietary preferences",
      "links": [
        {"memory_id": "memory-uuid-1", "basis": "direct"},
        {"memory_id": "memory-uuid-2", "basis": "direct"},
        {"memory_id": "memory-uuid-3", "basis": "direct"}
      ]
    }
  ]
}
```

无法形成有意义且满足约束的分组时明确推迟：

```json
{
  "result": "defer_split",
  "reason": "The remaining memories do not form coherent independent subjects."
}
```

`defer_split` 是语义决定；full/partial split 中的非法 ID、不完整覆盖、数量越界、新
subject 缺少 direct，或移出后某条 memory 不再有任何 active direct，仍是需要修复的输出错误。

#### 1.4.5 Subject summary refresh

输入仅包含 subject name，以及全部当前 active memories 的 content 和时间；不包含 subject
ID、memory ID、link basis 或旧 summary。输出整体替换 summary。该阶段是唯一生成 subject
summary 的阶段：

```json
{
  "result": "summary_refresh",
  "summary": "Complete summary supported by all supplied active memories."
}
```

同一 episode 的每个 target 使用独立输出和独立提交，最多并发 5 个。成功提交同时持久化
target 完成状态；恢复时只为尚未完成且仍 active 的 targets 再次请求该输出。

空白、超出字符硬上限或包含额外字段的结果按结构化输出规则修复；合法提交必须重新校验输入
快照仍与当前 subject 成员和 memory 内容一致。

## 2. 第一版正式版补充设计

正式版继续使用实验版核心表，并增加真实 Agent interaction 的可靠接收、buffer、
situation boundary 和 episode sealing。CLI、TUI 和 connector 不拥有另一套记忆数据。

### 2.1 正式版数据表示、缓冲与持久状态



#### 2.1.1 `users` 与正式 memory space

`users` 只保存建立默认 memory space 所需的稳定身份。第一版正式版按 user 创建默认
memory space；同一 user 的多个 streams 共享记忆，但各自维护独立 buffer。

#### 2.1.2 `interaction_streams`


| 字段                                | 含义                              |
| --------------------------------- | ------------------------------- |
| `stream_id`                       | interaction stream ID           |
| `memory_space_id`                 | 写入的 memory space                |
| `source_system`                   | connector 类型                    |
| `source_instance_id`              | connector 或设备实例                 |
| `source_stream_key`               | 宿主 session/conversation 的稳定 key |
| `parent_stream_id`                | 可验证 fork 的父 stream；普通独立 stream 为空 |
| `fork_parent_block_sequence`      | fork 发生在父 stream 的 block watermark；与 parent 同为空或同为非空 |
| `next_block_sequence`             | 下一个 stream block 顺序号            |
| `ingestion_version`               | 同 stream 写入 CAS version         |
| `created_at` / `last_received_at` | 系统时间                            |
| `metadata_json`                   | connector-specific metadata，可为空 |


来源身份使用以下唯一约束恢复：

```text
UNIQUE(source_system, source_instance_id, source_stream_key)
```



#### 2.1.3 `ingestion_receipts`

receipt 是不含正文的持久幂等账本，保存 stream、idempotency key、预处理前 canonical
payload hash、接收时间和本次写入分配的 block sequence 范围。

```text
UNIQUE(stream_id, idempotency_key)
```

相同 key 与相同 hash 重放时返回原接收结果；相同 key 携带不同 hash 时返回 conflict。
receipt 在对应 blocks sealed 后仍长期保留。

#### 2.1.4 `inbox_blocks`

SQLite durable inbox 直接保存尚未 sealed 的 user/assistant messages。每条 block 具有稳定
`block_id`、`stream_id`、全局 `stream_sequence`、`receipt_id`、role、assistant
`message_phase`、正文、预处理版本、内容 hash、`char_count` 和来源元数据。

一次 `add` 是幂等接收单位，但不是不可切分的 boundary 单位。boundary detector 按
stream sequence 检查 message，并且只允许在 `assistant final → user` 之间切分。因此一次
add 至少包含一轮完整交互；包含多轮交互时，可以在其内部形成 boundary。

宿主 transcript 中的 tool call、tool arguments 和 tool result 在 connector projection 时
跳过，不进入 receipt payload、durable inbox、开放 buffer、episode、overlap 或 provenance。
assistant message 必须显式保存 `message_phase = 'final'`，不能仅根据 role 猜测合法候选
边界。

#### 2.1.5 Boundary embedding 与 episode sealing 状态

`inbox_message_embeddings` 是可重建的临时数据，只为 user message 和 assistant message
保存 boundary embedding。它使用 `boundary` purpose 的 active model signature。

当 buffer 达到 message 或字符数阈值时，detector 按 2.2 节的规则选择合法边界。
Seal episode 在一个事务中：

1. 创建 `situational_episodes`；
2. 将 boundary 之前的 inbox blocks 按原顺序写入 `episode_blocks`；
3. 删除这些 inbox blocks 及其临时 boundary embeddings；
4. 保留 boundary 之后的 blocks 作为开放 buffer；
5. 记录 episode extraction 尚待处理。

正式版为 episode 增加 `stream_id` 和 `episode_ordinal`，并使用二者的唯一约束保证 stream
内顺序。Seal transaction 还必须从所属 memory space 原子分配单调递增且唯一的
`memory_space_write_sequence`；它只仲裁共享 memory store 的 stateful 写入顺序，不声称不同
streams 之间存在真实世界因果顺序。已经 sealed 的 episode 和 blocks 不允许改写或重新切分。
提取 overlap 直接读取同一线性 stream 的前一个 episode 尾部 message；overlap 只作为上下文，
不改变 provenance。

```text
UNIQUE(memory_space_id, memory_space_write_sequence)
```

是否整体 seal 取决于 soft/hard 阶段、assistant final 和候选最小容量，具体规则见 2.2。
`flush` 只处理请求 watermark 之前的内容，不永久关闭 stream。

#### 2.1.6 Flush 与恢复状态

`flush_requests` 保存 stream、幂等 key、请求时的 block-sequence watermark、创建时间和
完成时间。watermark 是请求被接受时该 stream 已分配的最后一个 block sequence，界定本次
请求覆盖的内容；它不是 worker 当前处理进度，之后到达的 blocks 不属于该次 flush。

```text
UNIQUE(stream_id, idempotency_key)
```

pending inbox blocks、未完成 flush request、未完成 episode extraction、pipeline 的临时
暂停或配置阻塞状态、累计淘汰计数和最后错误共同构成恢复与状态展示所需的事实状态。进程内
buffer、worker queue、LLM 调用上下文和普通日志都不是唯一副本。

#### 2.1.7 正式版写入事务

正式版在实验版事务之外增加：

1. **Receive interaction**：解析或创建 stream，验证幂等 key，连续分配 block sequences，
  写入 receipt 和 inbox blocks，并推进 `ingestion_version`。
2. **Seal episode**：写入 episode 和 episode blocks，为它原子分配
  `memory_space_write_sequence`，删除对应 pending blocks 和临时 embeddings，并建立待提取状态。
3. **Request/complete flush**：持久化 watermark；处理完成后标记 request，不影响之后到达
  的 blocks。

`add_into_buffer` 在 receive transaction 成功后即可返回，不等待 boundary detection、
memory extraction 或 subject 整理。同一 stream 的调用方按来源顺序提交，core 使用
`ingestion_version` CAS 防止并发覆盖；不同 streams 可以并行。

Public `add` 的成功只表示完整 add unit 已可靠写入 durable inbox，不表示已经产生 memory。
如果一个 add unit 自身无法放入空 buffer，receive transaction 必须拒绝它，不能先返回成功
再立即丢弃。

#### 2.1.8 Retention 与临时数据清理

正式版 episode、episode messages、memory/subject history、ended links、ingestion
receipts 和 domain operations 均不自动过期；显式删除 memory space 时清除整套数据。
正式版普通日志由宿主环境收集与轮转；其内容与持久运行状态的边界见 2.4.4。benchmark
artifacts 仍由 benchmark runner 管理。

boundary temporary embeddings 在 episode seal 的同一事务中删除。启动时立即删除所有
没有对应开放 buffer 的 orphan temporary embeddings，不设置清理延迟、周期或 batch size。

### 2.2 实时交互接收、分段与后台整理



#### 2.2.1 Public add 与来源投影

正式版 public `add` 接收宿主无关、按来源顺序排列的结构化交互，再委托内部
`add_into_buffer`；两者都只表示 durable buffer 接收，而不是 memory-store 写入。正常 Agent
集成由宿主生命周期 Hook 自动调用；第一版不向主 Agent LLM
暴露主动 add 工具，避免同一交互经 Hook 与 tool 重复写入。

每次 connector add 至少包含一轮 user message 到 assistant final message 的完整交互。
connector 只投影 user/assistant messages；tool call、arguments 和 result 在读取 transcript
时识别并跳过。Assistant 即使引用了 tool result，Memory Engine 也只使用 assistant message
中明确表达的正文，不能回读 tool result 作为证据。

Receive transaction 成功、message 可靠进入 durable buffer 后即可返回，不等待 boundary、
extraction、linking、review 或 split。幂等 key 与 canonical preprocessed payload hash 相同的
重放返回原 receipt；相同 key 不同 hash 返回 conflict。调用方负责按来源顺序提交同一 stream，
core 使用 `ingestion_version` CAS 防止覆盖；不同 streams 可以并行接收。

Boundary detection seal 出 situational episode 后，coordinator 调用与实验版相同语义的
`add_episode(memory_space_id, episode)`，即 `add_into_memory_store`。正式版只为这一共享核心
增加 overlap context builder；extraction schema、校验、修复重试、embedding、link/write、
maintenance 和终态失败语义不复制实现。内部调用在 episode 成功、合法零 memory 或持久化
终态失败后结束，但不改变此前 public `add` receipt 已成功返回的事实。

#### 2.2.2 确定性预处理

正式版不设置 `add_message_max`、`add_raw_total_chars_max`、
`add_preprocessed_total_chars_max` 或独立的 `message_chars_max`。每条 message 先执行统一、
版本化的确定性预处理：

- 对 API key、private key、session token、验证码等高置信度认证秘密格式执行确定性替换；
替换后的正文是进入 receipt hash、durable inbox、episode 和 provenance 的 canonical
source，原秘密不进入 SQLite、日志或备份。
- 第一版不尝试通用 PII 脱敏；无法高置信度识别的普通正文按本节裁剪规则持久化。
- tool call、arguments 和 results 已在 connector projection 阶段完全排除，不参与预处理。


| 配置项                  | 值      | 含义                              |
| -------------------- | ------ | ------------------------------- |
| `message_head_chars` | 17,000 | 超长 message 经过确定性预处理后从开头保留的原文字符数 |
| `message_tail_chars` | 14,000 | 超长 message 经过确定性预处理后从结尾保留的原文字符数 |


正文超过二者之和时，保留 head 与 tail，在中间加入固定简短 marker，例如
`……（中间内容已裁剪）……`。不设置 marker 长度配置。原始字符数、删除字符数、原内容
hash 和策略作为结构化 truncation metadata 保存；不设置
`truncation_marker_chars_max`。31,000 是由两项配置派生的保留预算，
不是另一项配置。

正式版 seal 后的 situational episode 不设 message 数或正文总字符硬上限。boundary
threshold 在通常情况下控制大小；无法合法切分的完整交互允许形成更大 episode。如果最终
无法装入 generation provider context window，则归入 `context_overflow`，该 extraction item
按永久单项失败处理，不能反向拒绝此前已成功接收的 add。

#### 2.2.3 Per-stream ring buffer


| 配置项                          | 值            | 含义                                                                    |
| ---------------------------- | ------------ | --------------------------------------------------------------------- |
| `stream_buffer_max_messages` | 256          | 单个 interaction stream 的 durable buffer 最多保留的 user/assistant message 数 |
| `stream_buffer_max_chars`    | 1,000,000 字符 | 单个 interaction stream 的 durable buffer 最多保留的 message 正文总字符数           |


两项分别判断，任一将被超过即从最旧的完整 add unit 开始淘汰，直到本次 add 可以整体放入；
绝不部分保留一个 add。如果新 add 自身无法放入空 buffer，则在 receive transaction 中整体
拒绝；其他已经成功接收的旧 add 仍可被淘汰。被淘汰或拒绝的内容不重试、不补写 memory，
必须记录 warning、终态原因和跨重启累计 counter。恢复后按来源顺序处理仍保留的内容。

不设置 warning/slow/reject 多级水位、全局 buffer 上限、active stream 并发、per-stream
pending add、ingestion lock timeout、backpressure 等待、connector add retry 或额外 LLM
并发配置。Embedding 和 SQLite 写等待复用实验版公共参数。

#### 2.2.4 Boundary detection

FluxFold 只为 user/assistant messages 生成 boundary embedding。合法候选边界只能位于
`assistant final → user` 的相邻消息之间，距离定义为 `1 - cosine_similarity`。


| 配置项                             | 值         | 含义                                        |
| ------------------------------- | --------- | ----------------------------------------- |
| `boundary_soft_message_count`   | 20        | buffer 达到该 message 数时开始尝试基于语义距离的 soft 切分  |
| `boundary_soft_chars`           | 20,000 字符 | buffer 达到该正文字符数时开始尝试基于语义距离的 soft 切分       |
| `boundary_hard_message_count`   | 48        | buffer 达到该 message 数时进入不要求最小语义距离的 hard 切分 |
| `boundary_hard_chars`           | 64,000 字符 | buffer 达到该正文字符数时进入不要求最小语义距离的 hard 切分      |
| `boundary_min_cosine_distance`  | 0.35      | soft 阶段接受候选 boundary 的最低相邻 message 余弦距离   |
| `boundary_episode_min_messages` | 4         | 候选 boundary 之前必须形成的最少 episode message 数   |
| `boundary_episode_min_chars`    | 500 字符    | 候选 boundary 之前必须形成的最少 episode 正文字符数       |


soft 和 hard 的 message/字符条件分别使用 OR。候选点之前必须同时达到 4 messages 与 500
字符；不满足者先排除，再选择剩余候选中 cosine distance 最大的一处。

- soft 阶段最大距离低于 0.35，或者没有候选满足 episode minimum 时，不切分并继续积累。
- 达到 soft threshold、完全没有语法上合法的 boundary、且 buffer 结束于 assistant final
时，整个 buffer seal 为一个 episode。
- hard 阶段有合格候选时忽略 0.35 并强制选择最大距离候选；没有合格候选时整体 seal。
- `flush` 可以把不足 minimum 的完整尾部整体 seal。整体 seal 没有实际候选点，不受
`boundary_episode_min_*` 限制。
- boundary 后的全部 messages 无条件留在开放 buffer；不设置 tail minimum。如果 tail
仍达到 soft/hard threshold，继续运行下一轮 detection。

不设置 `boundary_candidate_max`、`boundary_tail_min_messages` 或
`boundary_tail_min_chars`。当前 buffer 中全部合法候选参与比较，embedding 请求复用
`embedding_batch_size = 100`。Seal transaction 创建 episode、复制 boundary 前 messages、
删除对应 inbox rows 和 temporary embeddings，并保留 boundary 后的开放尾部。

#### 2.2.5 Extraction overlap


| 配置项                                           | 值        | 含义                                          |
| --------------------------------------------- | -------- | ------------------------------------------- |
| `extraction_overlap_message_max`              | 4        | extraction 最多取得的历史 user/assistant message 数 |
| `extraction_overlap_total_chars_max`          | 8,000 字符 | 全部 extraction overlap messages 的正文字符总预算     |
| `extraction_overlap_single_message_chars_max` | 4,000 字符 | 单条 extraction overlap message 的字符预算         |


从当前 episode 之前按时间倒序读取历史 messages，可以跨任意数量 episode，直到取得 4 条、
达到 8,000 字符或没有更早内容；不设置 `extraction_overlap_episode_max`。选中后恢复来源
时间顺序。单条超过
4,000 字符时在预算内约 50/50 保留 head/tail，marker 计入预算；这是固定算法，不增加配置。
Overlap 仅用于当前 episode 的指代和延续理解，不能独自产生 memory，也不能进入 provenance。

普通线性 stream 只从自身已 sealed 历史读取 overlap。新 stream 只有在 connector 能同时
验证父 stream 身份和精确 fork watermark 时，才保存 `parent_stream_id` 与
`fork_parent_block_sequence`，并允许首个 episode 在相同预算内从父分支 fork 点之前继承
尾部作为 overlap；共同历史不复制为新 episode、block 或 provenance。无法验证任一项时不
猜测父子关系、不继承 overlap，并记录 warning。

实验版 benchmark 不提供 overlap，正式 Agent 接入提供上述 overlap。两者只使用不同的
extraction input/context builder，共用同一 extractor prompt 契约、structured-output schema、
业务校验、retry、embedding 和失败处理，不维护两套 extractor 实现。

#### 2.2.6 Per-space coordinator、两阶段流水线与失败恢复

正式版为每个 memory space 独立推进 pending 工作，不设置覆盖全部 spaces 的顺序 worker。
不同 spaces 可以同时执行 boundary、extraction 和 stateful memory processing；FluxFold 面向
单用户本地服务，第一版不设置跨 space 的全局 generation/embedding provider 并发限制。
如果未来演进为多租户服务，再按部署容量引入全局 admission control。

每个 space 的 episode-to-memory pipeline 固定为两条单并发 lane：

1. **Extraction lane** 按 `memory_space_write_sequence` 一次处理一个 sealed episode，构造
   overlap、执行 extraction，并生成该批新 memory embeddings。episode N 完成 Stage A 后
   立即释放 lane，因此 N 等待或执行 Stage B 时，N+1 可以开始 extraction。
2. **Stateful lane** 也按 `memory_space_write_sequence` 一次处理一个 episode，依次完成候选
   召回、per-memory linking、原子 add commit、split/review，以及持久化 targets 的 subject
   summary refresh。一个 episode 的全部 maintenance 完成或进入终态后，下一
   episode 才能进入这条 lane；split 与 review 同时满足时仍按 1.2.4--1.2.5 的 split-first
   规则。

Extraction lane 本身就是并发上限 1，不增加 extraction concurrency 配置。Stage A 完成、
等待 Stage B 的 prepared results 只保存在 owner 进程内存，数量不设上限；这不会扩大正式
状态，也不改变顺序。进程退出时不持久化 prepared extraction，恢复后根据 immutable episode
和 pending 状态重新 extraction。后序 completion order 不能决定 link 顺序。

同一 user 的多个线性 streams 可以写入同一个默认 memory space，但各自的 blocks、buffer、
episode 和 overlap 始终隔离，不拼接成跨 stream episode。Seal 时分配的
`memory_space_write_sequence` 只为共享 subjects/memories 的 stateful mutation 选出确定顺序；
不会把 `A→B→C1→D1` 与 `A→B→C2→D2` 重排成一个 stream，也不声称两条分支之间存在因果
关系。

所有 memory-store mutation 共用同一 space 的写互斥边界：episode link/write/maintenance、
retrieval embedding rebuild/switch 和删除该 space。`clear_spaces` 使用覆盖所有 spaces 的
管理互斥。Public `add` 的 receive transaction 不属于该边界，不因后台 stateful lane 忙碌
而等待。


| 配置项                                     | 值     | 含义                                                        |
| --------------------------------------- | ----- | --------------------------------------------------------- |
| `memory_pipeline_failure_pause_seconds` | 300 秒 | 临时依赖错误在操作内重试耗尽且 provider 没有给出有效恢复时间时，写入流水线在下一次恢复探测前至少暂停多久 |


Generation 和 embedding provider 错误统一按 1.1.12 的 `error_class` 处理。
`transient_transport`、`rate_limited` 和 `service_unavailable` 在操作内 transport retry 耗尽
后使相关 memory space 进入临时暂停；带有明确恢复时间的 `quota_exhausted` 直接进入临时
暂停。SQLite 可重试
transaction/连接错误在 transaction retry 耗尽后执行相同策略。`rate_limited` 或
`quota_exhausted` 带有明确恢复时间时将 `pause_until` 设为该时间，其他临时错误设为当前时间
加 300 秒。暂停期间该 space 不开始新的 boundary、extraction、link/write、review 或 split，
其他 spaces 继续运行，search 仍读取已提交记忆。

正式版没有常驻 daemon 或 polling interval，因此 300 秒是最短暂停时间，不承诺到点立即
运行；到达 `pause_until` 后的下一次 worker 唤醒使用最早 pending item 探测恢复。成功即恢复，
再次发生临时错误则按新错误重新计算 `pause_until`。不另设连续失败阈值。

`authentication_or_configuration`、没有明确恢复时间的 `quota_exhausted` 和
`invalid_request` 使相关写入 pipeline 进入配置阻塞，不执行固定周期重试。配置变化或用户
显式执行 health check/恢复操作后，系统以最早 pending item 验证 generation/embedding
provider；成功后解除阻塞。不可恢复的数据库 schema、文件权限、磁盘空间或损坏错误采用
同一阻塞语义，不能伪装成 SQLite transaction 冲突反复重试。

`invalid_structured_output` 和 `incomplete_output` 完成 structured-output 修复后，受影响的
episode 或维护任务成为终态单项失败并继续；`context_overflow`、`policy_rejected` 和永久
输入错误直接成为终态单项失败。Episode 的确定性终态失败由共享 `add_episode` 核心在
`episode_extractions` 落库后抛出 `StageFailure`；重放读取同一失败，不再调用 provider。
以上单项失败不暂停 space，也不能反向拒绝已经成功 durable add 的交互。

正式版使用按需启动、队列排空后退出的本地 coordinator，不依赖 Hook、MCP、CLI 或 TUI 中
任一进程长期存活。Hook 完成 durable add、创建 flush request 或其他入口发现 pending 工作时，
只负责确保相关 space 的 coordinator 已被唤醒或启动，不等待记忆构建。多个本地进程通过
跨进程互斥保证**每个 memory space 同时最多一个临时 pipeline owner**；不同 spaces 可以由
相同或不同进程同时推进。owner 不是永久归属，其他进程仍可 add/search，队列排空后 owner
退出。具体互斥与进程启动机制由落地时确定，不引入 lease、heartbeat 或常驻 daemon。

启动 worker 失败或 worker 中途退出不会删除 pending 状态；下次 Hook、CLI、TUI、MCP 或
FluxFold 启动活动再次唤醒相关 space。进程崩溃后同样根据 SQLite pending state 继续，
进程内 queue、prepared extraction、buffer view 和 LLM 上下文都不是事实源。

#### 2.2.7 Flush 与读取新鲜度

`flush(stream_ref, idempotency_key)` 持久化当前 block-sequence watermark、唤醒 worker 后
立即返回，不等待后续阶段；它不写入交互，也不永久关闭 stream。不设置 flush timeout、
处理容量或 polling interval。调用方确需同步等待时，可为该次调用显式给出 timeout，系统
不提供全局默认值。

同一 flush 只覆盖 watermark 及其之前的 blocks；并发或稍后到达、sequence 更大的 add 留给
后续 boundary 或 flush。watermark 之前的完整尾部即使不足 episode minimum 也必须整体
seal 并进入 extraction。Flush 完成表示该范围内所有工作都已进入终态，终态可以是成功、
合法地产生零条 memory、任一终态单项失败或 buffer 淘汰；完成不承诺每个输入都产生 memory。
这些结果必须可以通过持久状态区分。

search 永远只读取已经提交的 memories/subjects，不等待 pending extraction。connector 在
durable buffer 写入成功后退出，不等待 memory write。

#### 2.2.8 Connector transcript 捕获


| 配置项                           | 值                      | 含义                                                |
| ----------------------------- | ---------------------- | ------------------------------------------------- |
| `transcript_record_max_bytes` | 8,388,608 bytes（8 MiB） | connector 流式读取 transcript 时允许单条原始 record 占用的最大字节数 |


connector 用持久 cursor 逐行流式读取 transcript，不把完整文件装入内存。单条原始 record
超过 8 MiB 时跳过并继续；末尾未写完的 JSONL record 不推进 cursor，下次 Hook 从该记录
开头重读。捕获是 best effort：读取或调用失败不做 connector 内部 retry，允许漏采且不阻塞
主 Agent。不设置 Hook/MCP request timeout、本地进程启动 timeout、transcript retry/backoff、
connector payload 总量、TUI page size 或 CLI 异步等待默认 timeout。

第一版 Claude Code connector 在每次正常 assistant final 后由 Stop Hook 触发，根据
`session_id`、`transcript_path` 等 payload 定位本轮新增交互。`stream` 表示一条线性对话
分支：宿主确认仍沿同一分支继续的普通 resume 复用原 stream；从共同历史产生不同后续的
fork/resume 必须建立新 stream。不同 streams 使用独立 buffers，但同一 user 的 streams 共享
默认 memory space。
Hook 可以是短生命周期子进程，search 通过由 Claude Code 管理的本地 stdio MCP server
暴露；两者共享 SQLite，不要求共享进程内存，也不构成 FluxFold 独立 daemon/service。

宿主能够报告 session 结束时，connector 必须先读取并 durable add 尚未接收的全部完整
transcript records，再对该 stream 创建覆盖最后一个已接收 block 的 flush request，使开放
buffer 的完整尾部立即进入处理。`/exit`、正常关闭等可观测退出采用此路径；强制杀进程、
宿主崩溃或断电没有 Hook 执行机会，不承诺即时 flush，但下次 FluxFold 活动必须根据 cursor、
durable inbox 和 pending flush 状态恢复并处理遗留尾部。未写完的 transcript record 不能作为
完整输入交给 extractor。

普通线性 resume 继续原 stream；fork、复制或宿主产生新的分支 identity 时建立新 stream，
仍写入同一 user 的默认 memory space。只有宿主提供的信息足以验证父 stream 与 fork point
时才保存 parent/fork watermark 并继承 extraction overlap；无法验证时建立无 parent 的新
stream、记录 warning，不能把两条分支交错写入原 stream。transcript 被截断、替换或
compaction 后，connector 不修改
已提交 episodes，也不重建历史，只继续接收能够可靠识别为新增的完整 records；无法判断时
记录 warning 并停止猜测。Connector 只解析经过验证的 transcript schema，检测到未知或不兼容
格式时明确失败，不宽松推断字段。cursor 编码、record identity 和轮转检测算法由实现确定。

### 2.3 正式版记忆检索与主 Agent 使用

正式版复用实验版 1.3 节的纯向量双通道 search、相似度门槛、top-k、attached entity、
去重和关系整理规则，不建立另一套检索实现。search 是主动检索通道，由主 Agent LLM 决定
何时调用和 query 内容；自动 add 与主动 search 职责分离。

Memory Engine core 不依赖 Claude Code、Codex 或 MCP 类型。connector 把 core 的 search
结构化结果转换为宿主工具结果。所有 streams 在同一 user 默认 memory space 中共享 active
memories/subjects；stream 只隔离开放 buffer 和 episode 顺序。Pipeline 暂停或存在 pending
extraction 时，search 仍立即返回当前已提交快照，不承诺读到最近已接收的交互。

第一版不提供 HTTP/gRPC 或远程 API，不承担远程认证、租户隔离、限流或分布式并发。CLI、
TUI、benchmark adapter 和 connector 都是 library 的薄适配层，不复制 search 业务逻辑。

### 2.4 正式版管理、接入与工程边界

本节只记录影响第一版范围和可靠性的工程规则。字段编码、命令名、输出 schema、异常类型、
cursor 算法、日志层级和测试 fixture 等实现细节在落地时依据本设计与通用工程原则确定。

#### 2.4.1 Claude Code connector 安装与配置所有权

Claude Code connector 与 FluxFold 主 package 一起发布，由 FluxFold CLI 提供显式安装、状态
检查、更新和卸载操作。安装会按宿主支持的方式注册 FluxFold 的 Hook、stdio MCP server 和
必要使用说明，因此属于对 Claude Code 配置或注册状态的明确修改。

- FluxFold 只合并和管理带有可验证所有权的自身条目，不覆盖、重排或删除其他 Hook、MCP、
plugin、skill 或用户配置。
- 修改任何已有配置文件前创建可恢复备份；配置损坏、格式未知或不能安全合并时停止修改并
给出诊断，不以重写整个文件作为恢复手段。
- 重复安装和更新保持幂等。卸载只删除能够确认由 FluxFold 创建的条目，不删除数据库、
memories、用户其他配置或无法确认所有权的相似条目。
- connector、Hook 与 MCP 通过已安装 distribution 的稳定入口启动，不依赖源码目录或当前
shell 工作目录。

第一版只支持经过端到端验证的 Claude Code Hook payload、transcript schema 和本地配置格式；
未知版本或格式明确报错。具体配置路径和合并算法由实现时依据所支持的 Claude Code 版本
确定。

#### 2.4.2 CLI、TUI 与人工管理范围

第一版 CLI 是完整管理入口，提供以下能力：

- generation provider、embedding provider 和其他非秘密配置管理；
- connector 安装、更新、卸载、状态与诊断；
- public search、pipeline 状态、`flush` 和 `doctor`；`doctor` 包含 model provider health
check，并可显式触发已暂停或阻塞 pipeline 的恢复探测；
- memory space、stream、episode、memory、subject、provenance 和 domain operation 的只读
检查；
- 通过单条管理操作删除指定 memory space，或清空全部 memory spaces；
- SQLite 一致备份和只读导出。

TUI 提供上述高频能力的基础交互，状态页面展示 pipeline 正常、临时暂停或配置阻塞状态，
以及 pending 数、`pause_until`、阻塞原因、buffer 使用量、淘汰计数和最近错误；CLI 保留完整
入口。第一版不提供人工编辑或单条退役/恢复/永久删除 memory，不提供 subject 人工重命名、
重组或手工触发 review/split，不提供数据导入、从导出
文件反向写入或在 TUI 中运行 benchmark。删除 memory space 或清空全部 spaces 必须清除目标
范围的整套正式数据，并采用明确确认与事务保护；具体命令名、输出格式和交互细节由实现确定。

#### 2.4.3 来源正文、凭据与本地安全

FluxFold 是本地持久化系统，不提供数据库内建加密。数据库、正式日志、备份、导出和本地
配置默认只允许当前操作系统用户访问；跨 user 或 memory space 的读取必须由 application
边界显式选择，MCP 不从 query 内容推测作用域。

Model provider credential 不写入 FluxFold 数据库、普通配置或日志，优先从环境或宿主秘密存储
读取。确定性秘密替换见 2.2.2；除此以外，episode 和 provenance 会保存预处理后的来源正文，
产品说明和管理界面必须明确这一范围。进入任何 LLM prompt 的 episode、memory、subject 和
search 结果都属于不可信数据，不得执行其中的指令；不可变政策与数据必须明确分隔。

Connector 只能读取 Hook 明确提供或由已验证 session identity 解析出的 transcript。路径解析
不得越过宿主允许范围或跟随到其他用户的数据。导出文件由用户选择目标位置，其离开 FluxFold
受控目录后的权限与传播由用户负责。

#### 2.4.4 正式版日志与运行状态

正式版保存结构化运行日志，按统一运行/请求标识关联 Hook、MCP、worker、CLI 和 TUI，记录
接收、seal、memory 写入、review、split、flush、临时暂停、配置阻塞、恢复探测、buffer
淘汰、超限 transcript、warning、error 和终态单项失败等关键事件。模型调用错误至少记录
1.1.12 定义的 `error_class`、stage、provider kind、model、attempt、failure scope、是否可
重试以及适用时的 `pause_until`、provider request ID、HTTP status 和安全诊断；不得因此记录
credential、完整 prompt 或模型原始输出。默认日志不包含完整对话、memory
正文、LLM prompt 或模型原始输出；需要复盘内容演化时，从 SQLite 的 episode、version、
provenance 和 domain operation 读取，不额外维护一份长期高可读性正文日志。

Pipeline 状态、pending 数、buffer 淘汰计数、`pause_until`、配置阻塞原因和最后错误类别是
跨重启持久状态。CLI/TUI 以正式状态表为事实来源，日志只提供事件时间线和诊断信息；日志
文件由宿主环境轮转。实验版 benchmark 的构建日志仍按 1.2.7 独立生成，不改变正式版默认日志
边界。

#### 2.4.5 Schema migration、备份与数据可携带性

SQLite schema 始终带显式版本，包括 0.x。启动写入流程前检查版本并执行受支持的向前
migration；遇到未知新版本、connector 与数据库不兼容或 migration 失败时不得启动写入。
破坏性或不可逆 migration 前自动创建 SQLite 一致备份，降级到旧 schema 不属于第一版承诺。

Prompt、LLM structured-output schema、提取规则或确定性预处理版本变化只影响后续输入，不
自动重写既有 memories。Embedding model 切换沿用 1.1.7 的全量重建与原子 active signature
切换规则。第一版提供 SQLite 一致备份和只读导出，不提供导入、双向同步，也不把导出文件
作为可写事实源。migration 的具体 SQL 步骤、中断恢复算法和备份文件格式由实现确定。

#### 2.4.6 Model provider、支持环境与第一版完成条件

第一版使用一个全局 active generation provider profile；memory extraction、linking、review
和 split 共享其连接配置，同时保留 1.1.11 已确定的各阶段模型参数。Retrieval embedding 与
boundary embedding 可以选择不同 embedding provider 和 model signature。Generation 与
embedding 配置都不提供 per-user 或 per-space 覆盖，也不要求运行中进程热加载；配置变更在
相关进程重启后生效，并允许下一次显式 health check 解除对应的配置阻塞。

第一版正式支持 Linux。macOS 和 Windows 只有在 connector、路径、文件权限、跨进程互斥和
终端行为均有对应测试后才声明支持。正式版完成条件是 Claude Code connector 可以完成安全
安装、交互捕获、session 结束 flush、异常恢复、记忆构建和 MCP search；CLI/TUI 可以检查
状态、诊断故障、备份及删除数据；进程崩溃、model provider 临时暂停和配置阻塞解除后可以
从持久状态恢复。Benchmark 分数是实验结果，不作为正式版发布门槛。

## 3. 未来方向

以下内容不属于第一版承诺；只有 benchmark、profiling、真实 Agent 使用或维护成本提供可
复现证据后，才进入当前设计和集中配置。

### 3.1 检索与存储扩展

- 实验 BM25、关键词或其他词法召回，以及向量/词法候选池、融合权重、RRF、重排、分词和
BM25 参数；当前所有检索保持纯向量。
- 对比 subject name 与 name+summary embedding，评估 summary 提供的信息和陈旧影响。
- 评估最低相似度、top-k、attached entity 数和最终上下文容量对不同查询类型的影响。
- 只有精确扫描出现可复现性能瓶颈时，评估 SQLite vector extension、ANN 或外部向量索引，
并先定义一致性、切换与重建规则。
- 只有 memory-to-memory typed edges 和不定深度多跳成为核心能力且关系查询出现瓶颈时，
才评估图数据库；当前 subject-memory 与 provenance 继续使用 SQLite 关系表。



### 3.2 Memory Engine 行为演进

- 当标注或端到端结果持续出现过切、欠切或成本问题时，对比其他 embedding/boundary 参数
与 LLM boundary detection。
- 对候选召回参数、Subject/Memory 通道和主动 association search 做消融，评估漏链、错链、
link basis 与推理成本。
- 如果写入顺序持续造成旧 memory 缺少后来才成立的 subject link，再评估有限 link 回填。
- 根据 review/split 的去重、冲突、summary、碎片化、重叠和成本结果，实验新的 prompt、
schema、触发条件、provenance 范围、拼接粒度和分组规则。
- 只有公开 search 出现低相关结果或上下文挤占时，实验最终字符预算和裁剪策略。



### 3.3 Scope、connector 与部署扩展

- 在 user 默认 space 之外评估 workspace/project spaces；采用前定义 scope 解析、权限、迁移
和跨-space retrieval。
- 后续 connector 至少面向 Codex、Pi、OpenClaw 和 Hermes Agent。实现第二个宿主后，根据
重复接线成本决定是否提炼 connector adapter contract 或拆分 package。
- 只有宿主退出导致长期积压、多个进程必须可靠协调、需要远程访问或集中管理时，才评估
local daemon、service、HTTP/gRPC、认证和多租户能力。



### 3.4 人工可读数据与工程演进

- 在第一版 SQLite 只读导出的基础上，只有明确所有权、冲突和事务语义后才考虑导入或从
Markdown/其他格式反向写入；系统不同时维护两个可写事实源。
- 当 UI/connector 依赖显著增加、组件需要独立发布，或 package 边界造成持续成本时，再评估
optional extras、多 distributions、workspace 或独立 connector package。
- native extension、任务编排器、pre-commit、更多测试工具和发布基础设施只在已有工具出现
可观察缺口时引入。
