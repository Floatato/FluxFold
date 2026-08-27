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
| `summary`                   | 当前 summary                            |
| `lifecycle_status`          | `active` 或 `retired`                  |
| `new_memory_count`          | 上次成功审核或分裂后新增的 memory 数量               |
| `summary_revision`          | summary 每次 append 或重写时递增的 CAS version |
| `created_at` / `updated_at` | 系统时间                                  |
| `retired_at`                | subject 被替换的时间，可为空                    |
| `retired_by_operation_id`   | 产生替换的 subject split operation，可为空     |


subject name 在一次 subject 生命周期内保持不变。Subject 关联记忆审核只重写 summary；
Subject split 可以完整替换原 subject、从中拆出若干更具体的 subjects 并保留原 subject，
或者明确推迟本次分裂。只有完整替换才退役原 subject；partial split 保留其 ID 和 name。

每当新 memory 建立指向已有 subject 的 link，系统立即把 memory content append 到 summary，
递增 `new_memory_count` 和 `summary_revision`。本批新建 subject 使用 Subject linking 给出的
完整初始 summary，初始 `new_memory_count` 为零，不再重复 append 同批 memory。审核成功后，
系统整体替换 summary，并把 `new_memory_count` 置零。共享 memory 被其他 subject 的 review
修改或退役后，本 subject 的 summary 可以暂时包含旧信息；直到本 subject 以后因新增 link
正常触发 review 时，才按届时的全部 active memories 重写。

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
新行，不复用旧行。`direct` 表示 memory 直接描述 subject 本身或其核心事实直接属于该
subject；`contextual` 表示 memory 不直接描述 subject，但会具体影响、约束、更新或解释
该 subject 下的信息，遗漏它可能实质性损害未来回答。每条新 memory 至少建立一个
`direct` link，且可以有多个 `direct` link；全部 active subject links 合计不得超过 5。
`contextual` link 按需建立，不表示较低优先级。
SQLite 使用 `CHECK (link_basis IN ('direct', 'contextual'))` 约束取值。

`link_basis` 在一次 link 生命周期内保持不变。需要重新分类时关闭旧 link，并以新的
`link_basis` 建立新 link。Subject review 读取当前 subject 下每条 memory 的
`link_basis`，但不修改它；两类 link 都参与 review、summary 和 search，第一版不根据
该字段过滤或调整检索分数。

任何流程新建 memory link 时，如果 LLM 判断候选 subjects 中一个是另一个的语义具体化，
只建立指向语义正确且更具体 subject 的 link。该规则不禁止一条 memory 同时关联两个没有
包含关系的领域。包含关系只由 LLM 根据当前输入判断；程序不保存 subject 层级、不主动维护
包含关系，也不对一般性的语义包含关系作校验。Partial split 中原 subject 与本次新 subjects
之间的关系由操作类型直接确定，按下述 split 规则关闭旧 links。

Subject 分裂时：

- full split 关闭全部输入 memories 指向原 subject 的 links；partial split 只关闭被移入本次
新 subjects 的 memories 指向原 subject 的 links；
- 保留这些 memories 指向本次 split 范围之外其他 subjects 的 links；
- 对每条新建结果 link 重新判断 `link_basis`，不能直接继承原分类；partial split 中未移出的
原 links 保持不变；
- 直接描述结果 subject 的 memories 用于确定分组与命名，contextual memories 只关联到
仍存在具体联系的结果 subject；
- 每个新 subject 至少有一条 `direct` link，事务结束后每个 active memory 仍至少有
一条 active `direct` link。

memory 退役时关闭它的所有 active links，但保留 link rows 作为历史记录。

#### 1.1.7 Embedding



##### Model signature 与用途

`embedding_model_signatures` 保存 provider、model ID、revision、dimension、dtype、
normalization 和 query/document encoding mode。配置不同即视为不同 signature。

模型不在本设计中硬编码。每个 memory space 按用途选择 active signature：

- `retrieval`：memory、subject 和 query 检索；
- `boundary`：正式版的 message boundary detection。

同一次相似度计算不得混用不同 signature。模型切换时，先生成对应用途的全部必要
embedding，再原子切换 active signature。

##### `memory_embeddings`

每个 active memory 的 latest version 在 retrieval signature 下恰好有一个 embedding。
embedding 输入只包含 memory content。memory 被修改时生成新 embedding；memory 退役时
删除其检索 embedding，正文仍可用于重建。

##### `subject_embeddings`

每个 active subject 同时维护：

- `name`：只编码 subject name；
- `name_summary`：编码 subject name 与当前 summary。

表以 `(subject_id, embedding_kind, model_signature_id)` 唯一标识一条 embedding，并保存
输入文本的 hash。name 变化时更新两种 embedding；summary 变化时只更新
`name_summary`。公开 search 默认使用 `name`，写入阶段的 Subject 候选通道始终使用
`name`。

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
限流或服务不可用在 item retry 耗尽后不写入该状态，而是保留未完成 episode 供显式恢复。

##### `domain_operations`

`domain_operations` 是 append-only 的结构化审计记录。operation 表示一次业务原子变化，
而不是每个单表写入。第一版至少包含：

- `add_episode_memories`
- `review_subject`
- `split_subject`
- `retire_memory`

operation 保存 memory space、actor、规则或模型配置、简短 reason 和提交时间。
`domain_operation_effects` 统一记录受影响的 memory、memory version、subject 或 link 及其
effect type。link row 同时通过 open/close operation ID 保留关系变化来源。

#### 1.1.9 原子事务与并发校验

以下变化分别作为一个 SQLite transaction 提交：

1. **Add episode memories**：写入 extraction completion、memory units、latest versions、
  provenance、`latest_source_at`、embeddings、subjects、links、初始 summaries 和
   operation effects。
2. **Review subject**：写入全部 memory 新版本和 provenance，退役指定 memories 并关闭其
  active links，重新计算受影响 memory 的 `latest_source_at`，更新 embeddings，整体替换
   当前 subject summary，更新 summary embedding，将新增计数置零，并记录 operation
   effects。
3. **Split subject**：full split 创建结果 subjects、summaries 和 embeddings，关闭原
  subject 的全部 links，建立结果 links 并退役原 subject；partial split 创建新 subjects、
   summaries 和 embeddings，只关闭移出 memories 指向原 subject 的 links，建立新 links，
   更新原 subject summary 并保持其 active。两种成功结果都将涉及的 active subjects 的新增
   计数置零并记录 operation effects。
4. **Switch embedding model**：新 signature 的全部必要 embeddings 准备完成后，切换对应
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
| append 后存储的 subject summary   | 不校验总长度                                 |
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


所有阶段均不设置 `max_output_tokens`。普通运行不设置 seed；generation provider 支持 seed
时，benchmark 使用 `42`。不设置全局或分阶段 LLM 并发上限，也不默认设置
requests-per-minute 或 tokens-per-minute；即默认不设置 `requests_per_minute` 或
`tokens_per_minute`。Generation provider/deployment profile 可按真实外部配额覆盖。不支持 temperature
的 generation provider profile 省略该参数。Memory Engine 不设置分阶段 request timeout、
阶段 deadline，也不在 generation provider 外再包一层 timeout。连接和单次请求 timeout 由
generation provider 的 transport 负责；timeout 归一化后在 provider 内执行下述 transport
retry，耗尽后才把错误返回 Memory Engine。

本文用 model provider 统称外部模型服务；调用 extraction、linking、review 和 split 的服务
称为 generation provider，生成 retrieval 或 boundary vector 的服务称为 embedding provider。
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
| `embedding_request_timeout_seconds`         | 60 秒        | 单次 embedding provider 请求的 timeout               |
| `embedding_transport_max_retries`           | 5           | embedding 初次传输失败后的额外重试次数                        |
| `embedding_max_concurrency`                 | 4           | 同时执行的 embedding provider 请求上限                   |
| `exact_scan_batch_rows`                     | 8,192       | NumPy 精确扫描时每批从 SQLite 读取并计算的 embedding 行数       |
| `sqlite_busy_timeout_ms`                    | 5,000 毫秒    | SQLite 遇到锁竞争时等待锁释放的最长时间                         |
| `sqlite_transaction_max_retries`            | 5           | 初次 SQLite transaction 冲突后的额外重试次数                |
| `sqlite_transaction_retry_initial_seconds`  | 0.01 秒      | SQLite transaction retry 的指数退避初值                |
| `sqlite_transaction_retry_multiplier`       | 2           | SQLite transaction retry 每次增长的倍数                |
| `sqlite_wal_autocheckpoint_pages`           | 1,000 pages | SQLite WAL 自动 checkpoint 的 page 阈值              |
| `memory_space_write_concurrency`            | 1           | 同一 memory space 同时提交正式状态写入的数量上限                 |
| `subject_maintenance_concurrency_per_space` | 1           | 同一 memory space 同时执行 Subject review/split 的数量上限 |


memory 或 subject 创建、相关文本更新时立即计算对应 embedding。一个逻辑操作同时产生多个
文本时按 100 个一批合并请求；单个文本的批次就是 1。全部必要 embedding 在事务外生成
成功后，与正式内容和关系原子提交，不能暴露缺少当前 embedding 的 active 对象。Embedding
provider 只有归一化为 `transient_transport`、`rate_limited` 或 `service_unavailable` 的错误
才执行额外 5 次重试，并复用 generation transport retry 的退避参数；配置、权限、硬配额和
非法请求错误遵守上表的阻塞规则。

SQLite 固定使用 WAL、`synchronous=NORMAL`。事务冲突最多额外重试 5 次，使用 full
jitter，不设置退避最大时间。同一 memory space 的正式写入并发为 1，review/split 维护
并发也为 1；事务 retry 不设置退避最大时间。不设置跨 space 的全局维护并发或单次 add
的维护操作数量上限。

#### 1.1.13 Public library 与 memory-space 管理边界

实验版 public library 提供以下能力：创建或打开 memory space、删除指定 memory space、
清空全部 memory spaces、向指定 space 添加一个已规范化 dataset episode，以及在指定
space 中检索记忆。它还提供检索 embedding 的全量重建：先为全部 active memory latest
content 和 active subject 的 name/name-summary 生成新 signature 下的向量，再在一个事务
中校验来源快照并切换 active retrieval signature。删除指定 space 和清空全部 spaces 都能通过单条管理命令完成；它们是
memory-space 级管理操作，会清除目标 space 的整套数据，不改变普通流程中 episode、历史
版本和 domain operation 的不可变约束。

Dataset adapter 只把来源格式转换为统一 episode。Memory Engine 不直接接受
LongMemEval-S 或 LoCoMo_refined 的原始 record，adapter 也不实现或复制 extraction、
Subject linking、Subject review、Subject split、正式写入或 search 逻辑。

同一规范化来源身份和相同 canonical payload 的重复提交是同一次逻辑 add 的安全重放，
不重复调用模型或写入；同一来源身份对应不同 canonical payload 时拒绝覆盖原 episode。
一次 add 不暴露部分提交的正式 memory 或 subject 状态。

Public search 的结构化结果是 library 的事实输出；供主 Agent 使用的文本由同一结构化结果
独立渲染，不得改变其中的对象、关系或顺序。同步或异步调用形式、准确函数签名、具体返回
类型和异常类由实现根据所选运行框架确定，不构成第一版领域设计约束。

### 1.2 记忆添加、关联与整理



#### 1.2.1 实验版入口与总体流程

实验版每次 add 一个已经划定边界、不可再切分的 dataset session；LongMemEval-S 的一个
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

##### Add 流程

规范化并持久化 episode 后，完整流程为：

```text
episode
→ memory extraction
→ 对每条新 memory 召回 subject/memory 候选
→ 把本 episode 的全部新 memories 作为一个组织批次执行 Subject linking
→ 原子写入 memory、provenance、embedding、subject、links 和 operation
→ 触发并完成已排定的 Subject review / Subject split
```

LLM 与 embedding 在事务外运行。`episode_extractions` 以 episode input hash 和 extractor
配置签名保证一次逻辑完成只提交一次。Extraction 明确返回“没有有价值的 memory”时也必须
写入 completed 状态；缺失、畸形或校验失败的输出不能产生该状态。

#### 1.2.2 Memory extraction

`memory extractor` 从当前 episode 提取 `0..N` 条自包含 memory unit。是否值得提取以
“遗忘是否会明显损害未来交互”为核心判断：会影响连续性、个性化、任务继续、后续指代、
状态变化或重要决定的信息应当提取；寒暄、可重新生成的通用知识、未被接受的建议、机械
操作过程和没有未来用途的重复内容通常不提取。

一条 memory 表达一个可以独立检索、更新或失效的完整对象：明确主体，加一个状态、事件、
决定或目标，以及理解它所必需的时间、条件、原因或直接结果。不同主体、不同生命周期、
不同时间范围或可以分别完成的事项应拆开；同一不可分割事实的条件和直接结果应保留在一起。

memory 必须脱离 episode 后仍可理解，消除含糊代词并明确关系双方。extractor 只能重组
episode 明确支持的信息，不能推测动机或因果；Assistant 建议只有被 User 明确接受后才能
写成已确认方案。外部事实的来源归属、不确定性、计划/进行中/完成/失败/取消等状态必须保留。
同一 episode 内的明确纠正以最终状态为准；跨 episode 的重复、冲突和状态变化交给 Subject
review 处理。

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

被动候选召回以新 memory content 为 query。开启可选主动关联检索后，LLM 还可以最多一次
调用 `association_search(query)`；主动 query 应表达可能的影响方向，而非复述新 memory。
两种召回使用同一参数：


| 配置项                                        | 值    | 含义                                             |
| ------------------------------------------ | ---- | ---------------------------------------------- |
| `subject_top_k`                            | 12   | Subject 通道最多保留的 subject 数                      |
| `subject_min_cosine_similarity`            | 0.25 | Subject 通道允许候选进入结果的最低 query-subject name 余弦相似度 |
| `subject_attached_memory_k`                | 1    | 每个 Subject 通道候选附带的关联 memory 数                  |
| `memory_top_k`                             | 24   | Memory 通道最多保留的 memory 数                        |
| `association_memory_min_cosine_similarity` | 0.35 | Memory 通道允许候选进入结果的最低 query-memory 余弦相似度        |
| `memory_attached_subject_k`                | 1    | 每个 Memory 通道候选附带的关联 subject 数                  |
| `association_search_max_calls`             | 1    | 一次关联决策中最多执行的主动 association search 次数           |
| `memory_link_preferred_min`                | 1    | Prompt 建议一条 memory 通常至少链接的 subject 数           |
| `memory_link_preferred_max`                | 4    | Prompt 建议一条 memory 通常最多链接的 subject 数           |
| `memory_active_subject_link_max`           | 5    | 一条 memory 可以同时拥有的 active subject link 硬上限      |


Subject 通道按 query 与 active subject name embedding 取前 12 个，再删除相似度低于 0.25
的 subject。每个保留 subject 附带其 active memories 中与 query 最相似的 1 条；没有
linked memory 时只提供 subject。

Memory 通道按 query 与 active memory content embedding 取前 24 条，再删除相似度低于
0.35 的 memory。每条保留 memory 附带其 active subjects 中 name embedding 与 query 最
相似的 1 个；没有 subject 时只提供 memory。

两个通道完成后按真实 subject-memory 关系去重和整理，不计算融合分数、不建立更大的中间
候选池、不重排或再次淘汰。所有检索只使用向量相似度；第一版不使用 BM25、全文或关键词
匹配。

程序在 extraction 后为本 episode 的全部新 memories 分配仅用于本次组织批次的临时
`memory_ref`。各 memory 的候选可以分别召回，但 LLM 必须在一个最终输出中给出整个批次的
全部 memory–subject links，以及全部待新建 subjects 的临时引用、name 和 summary。同批
memories 不是依次写入；任何新 subject 都在完整批次校验通过后才取得正式 ID。LLM 输出和
必要 embeddings 全部准备成功后，整个批次与 memory、provenance 和 extraction completion
原子提交。

LLM 判断应链接哪些已有 subject、是否新建 subject，以及每条 link 的 `direct` 或
`contextual` basis。直接描述 subject 核心事实使用 `direct`；只有遗漏某条非直接描述的
memory 会实质损害未来回答时才建立 `contextual`。宽泛常识联系不足以建立 link。Prompt
建议每条 memory 通常建立 1--4 个 links，硬校验要求每条 memory 至少一个 direct link，
全部 active links 不超过 5。已有 subject 与其语义具体化 subject 同时成为候选时，LLM 只
选择语义正确且更具体的一个；程序不校验或持久化这种包含关系。

每当新 memory 建立指向已有 subject 的新 active link，立即把 memory content append 到该
subject summary，递增 `new_memory_count` 和 `summary_revision`，不调用 LLM 重写。本批新建
subject 的初始 summary 由 Subject linking 输出，只能总结本批实际链接到它的 memories，
其 `new_memory_count` 从零开始。已有 subject append 后的当前 summary 不执行长度校验。

#### 1.2.4 Subject review

一个 subject 自上次成功 review 或 split 后，每新增 8 条具有新 active link 的 memory，
触发一次 review：


| 配置项                                   | 值   | 含义                                                  |
| ------------------------------------- | --- | --------------------------------------------------- |
| `subject_review_new_memory_threshold` | 8   | 自上次成功 review/split 后新增到该 subject 的 memory 数量触发阈值    |
| `review_provenance_memory_max`        | 8   | 一次 Subject review provenance request 最多指定的 memory 数 |
| `memory_provenance_episode_max`       | 6   | 一个 memory version 最多关联的来源 episode 数                 |


direct/contextual 使用同一计数；同一 memory-subject 对只计一次。review 读取该 subject
当前全部 active memories，而非仅新增的 8 条。成功后计数清零，失败不清零。没有独立的
summary review，也不按 summary 长度触发 review。

审核输入包含 subject name，以及每条 active memory 的 ID、content、相对当前 subject 的
link basis 和时间等必要元数据。审核允许保留、修改和全局退役 memory，不允许新建、拆分
memory 或调整 links。修改创建新 memory version；退役关闭该 memory 的全部 active links，
使其退出候选、review、split、summary 和公开 search，但保留历史正文、provenance、links
与 operation。

审核进行一至两次结构化输出。仅在 content 和元数据不足以解决重复、冲突、纠正、状态
变化或信息归属时，首轮可以请求 provenance；一次最多请求 8 个不同 memory IDs，系统返回
这些 memories 当前 provenance 涉及的全部 episodes，不设置
`review_provenance_episode_max`。获得来源后
必须输出最终结果，不能再次请求。

最终结果列出 memory updates、memory retirements 和完整的新 subject summary。未列出的
memory 保持不变。update 可以替换 content、provenance 或两者，但必须至少改变一项；提供
provenance 时，它表示包含 1--6 个不同有效 episode IDs 的完整替换集合。超过 6 不能截断；
无法由至多 6 个来源准确支持的合并不得执行。

只允许拼接共同描述同一个、不可独立更新事实的 memories。不同但相关、可以分别变化的事实
继续分开。审核不能默认新事实覆盖旧事实，也不能仅因时间较早退役；无法解决的冲突应保留，
并在 summary 中准确表达不确定性。

LLM 先在逻辑上应用 updates 与 retirements，再根据最终 active memories 生成不超过 200
words、字符硬上限 2,000 的 subject summary。系统整体替换当前 subject summary，不修改
subject name。共享 memory 的内容或生命周期修改立即全局生效；其他 subjects 的 summary
允许暂时陈旧，直到它们以后因新增 link 正常 review。

#### 1.2.5 Subject split

split 只在新 memory 建立指向某 subject 的 active link 时检查。共享 memory 因其他 subject
review 而更新或退役、或没有新增 link 的其他容量变化，不触发检查。


| 配置项                                          | 值         | 含义                                                                  |
| -------------------------------------------- | --------- | ------------------------------------------------------------------- |
| `subject_split_memory_count_threshold`       | 32        | subject 的 active memory 数量触发 split 的阈值                              |
| `subject_split_total_memory_chars_threshold` | 20,000 字符 | subject 下 active memory latest content 总字符数触发 split 的阈值             |
| `subject_split_result_subject_min`           | 2         | full split 的新 subjects 数或 partial split 的原 subject 加新 subjects 总数下限 |
| `subject_split_result_subject_max`           | 5         | full split 的新 subjects 数或 partial split 的原 subject 加新 subjects 总数上限 |
| `subject_split_result_min_memories`          | 2         | 每个新建结果 subject 至少必须包含的 memory 数                                     |
| `subject_split_result_target_memory_max`     | 20        | 每个新建结果 subject 的目标 memory 容量上限                                      |
| `subject_split_memory_membership_max`        | 2         | 一条输入 memory 最多可以归属的本次新建 subject 数                                   |


active memory 数达到 32，或 latest contents 总字符数达到 20,000，即满足触发条件；字符
条件不要求另一个最小 memory 数。两个数值都是触发整理的软阈值；实验版不限制单个 subject
最终关联的 memory 数量或总字符数，不设置硬容量、动态阈值、冷却期或封存状态。每次又有
memory link 到已达到任一阈值的 subject 时，都重新尝试 split。

split 的结构化结果只能是 `full_split`、`partial_split` 或 `defer_split`：

- **full split** 创建 2--5 个更具体且可独立理解的新 subjects。原 subject 的每条 active
memory 至少进入一个、最多进入两个新 subjects；原 subject 被完整替换并退役。
- **partial split** 创建 1--4 个更具体的新 subjects，同时保留原 subject 的 ID、name 和
active 状态。LLM 只列出新 subjects 的 name、summary，以及应移入它们的 memory IDs 和
link basis，并给出移走这些 memories 后原 subject 的完整 `remaining_summary`。程序取
所有被列出 memory IDs 的并集，关闭它们指向原 subject 的 links；未被列出的 memories
继续留在原 subject。移出集合必须非空且不是原集合，原 subject 至少保留一条 memory；
否则结果应表达为 full split 或 defer split。
- **defer split** 表示当前 memories 无法形成满足约束且具有实际组织意义的分组。它是合法
业务结果，不修改 subjects、memories、links 或计数，并记录 warning；下次又有 memory
link 到原 subject 且容量仍达到阈值时再次尝试。它计入 Subject split 触发次数，不计入
benchmark 失败样本数。

每个新 subject 至少包含 2 条 memories，目标不超过 20 条，并且至少有一条 `direct` link。
一条 memory 在本次新 subjects 中最多出现两次；partial split 中出现在任一新 subject 的
memory 到原 subject 的 link 由程序关闭，但它指向 split 范围之外其他 subjects 的 links
保持不变。不设置结果 subject 的字符目标、总 link 倍数或整体重叠率上限。

分组依据是未来是否需要独立检索、更新和增长，而不是平均分配数量。结果 name 应保留原
主体锚点并表达具体领域、项目模块、事件阶段或人物关系；不得使用没有语义边界的 Other、
Misc 等名称。结果 subjects 存在语义包含关系时，LLM 只把 memory 分配给语义正确且更具体
的一个；程序不校验或维护这种包含关系。所有新 links 的 `link_basis` 重新判断，不能继承
原值；contextual memory 只进入仍有具体联系的新 subject。

例如，原 subject `Mike` 中只有一组 memories 足以形成 `Mike's dietary preferences` 时，
partial split 把这组 memories 移入新 subject，其余无共同具体领域的 memories 继续留在
`Mike`。被移走的 memory 不再同时 link 到 `Mike`，但可以继续 link 到与本次 split 无关的
其他 subjects。

full split 和 partial split 都在一个事务中提交新 subjects、summaries、embeddings 和 link
变化，memory content 与 provenance 不变。成功后所有结果 active subjects 的新增计数归零；
partial split 同时整体替换原 subject summary。Review 与 split 同时满足时先执行 split：
full 或 partial split 成功后无需立即 review；defer split 后仍执行已经达到触发条件的 review。

JSON、schema、字段类型或非法 ID 错误归入 `invalid_structured_output`，按 1.1.12 的规则
立即反馈并修复。结果虽然符合 schema 但违反 ID 覆盖、结果数量、最少 memories、direct
link 或 membership 等业务不变量时也归入同类；不能静默改写为 defer split。LLM 判断不
存在有意义的合法分组时应直接输出 `defer_split`。

#### 1.2.6 Benchmark 执行与可复现性


| 配置项                                            | 值       | 含义                                                                |
| ---------------------------------------------- | ------- | ----------------------------------------------------------------- |
| `benchmark_memory_space_build_concurrency`     | 10      | benchmark 同时构建的独立 memory space 数                                  |
| `benchmark_extraction_concurrency_per_space`   | 10      | 同一 benchmark memory space 同时执行的无状态 session extraction 数           |
| `benchmark_search_concurrency`                 | 5       | 同时执行的 benchmark search/QA 样例数                                     |
| `benchmark_seed`                               | 42      | generation provider 支持 seed 时 benchmark 使用的固定 seed                |
| `benchmark_memory_space_build_timeout_seconds` | 7,200 秒 | 构建一个 memory space 的总 timeout                                      |
| `benchmark_search_sample_timeout_seconds`      | 300 秒   | 一条 benchmark search/QA 样例的 timeout                                |
| `benchmark_failure_max_retries`                | 2       | 单项操作内 transport/transaction retry 耗尽后，benchmark 对该完整 item 的额外重试次数 |
| `benchmark_checkpoint_interval_items`          | 1       | 每完成多少个项目保存一次 checkpoint 和结果                                       |


LoCoMo 的 10 个 conversations 分别建立 10 个 memory spaces，可同时构建。LongMemEval 的
500 个 evaluation instances 分别建立独立 memory space；不同 instance 的 haystack 不能
合入同一 space。

同一 space 最多并行执行 10 个无状态 session extractions，不设置
`benchmark_extraction_prefetch_per_space`、等待队列长度或已完成但待提交结果上限。完成
顺序可以不同，但后续候选召回、linking、正式
写入、review 和 split 必须按 session `source_sequence` 串行提交；后序结果不能越过仍在
处理或仍可重试的前序 session。前序 session 已成为终态单项失败后，记录失败并继续推进
来源顺序。不同 spaces 的有状态流程可以并行。

不设置整次 benchmark 总 timeout。每完成 extraction、space ingestion 或一条 QA 都原子
保存 checkpoint。只有 `transient_transport`、`rate_limited`、`service_unavailable` 或
可重试的 SQLite transaction/连接错误，才在所属操作内重试耗尽后由 benchmark 额外重跑
完整 item 2 次；这两次 item retry 不替代也不增加单次操作的 transport/transaction retry
上限，并继续遵守 1.1.12 的退避和有效 `Retry-After`。模型给出的结构和业务均有效但错误的
答案不重试。一次 full 或 sample 脚本只执行一个 run；需要重复实验时调用方使用不同 run
目录手工重复执行。runner 不内置重复次数，也不跨 run 计算均值或标准差。

每套数据集提供 `build`、`answer`、`score` 三个独立 stage，并分别提供 full 与 sample
薄脚本，共六个可直接通过 `python -m benchmarks.scripts.<stage>_<mode>` 运行的模块。
build 只构建数据库和写入审计产物；answer 从同一 run manifest 和数据库执行 public
search 并生成官方字段形状的 predictions；score 独立读取 predictions 生成逐题结果和汇总。
stage 之间使用不可变 manifest 校验数据文件 hash、选择范围、配置签名、
embedding signature 和 seed。manifest 记录 build 使用的 generation model，供复盘写入侧；
answer 和 score 各自读取独立的 generation provider 配置，不要求与 build model 相同。
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
`API_KEY`、`BASE_URL`）；retrieval embedding 使用独立 embedding model 配置。
LongMemEval 输出 `question_id`/`hypothesis`，按其公开 rubric 进行等价答案 LLM 判断。
LoCoMo_refined 输出 `qa_id`/`predicted_answer`，使用
根据公开指标说明独立实现的严格 LLM judge、token F1 和 BLEU-1；多个合法 reference 取最佳
匹配。本仓库不导入或调用 LoCoMo_refined 的非商业许可 evaluator 源码。

上述临时错误在 item retry 耗尽后暂停整个 benchmark 记忆构建流水线；`rate_limited` 或
`quota_exhausted` 给出明确恢复时间时暂停到该时间，否则进入临时暂停并等待下一次显式恢复
探测，恢复后从最早未完成项目继续。`authentication_or_configuration`、没有恢复时间的
`quota_exhausted`、`invalid_request` 或不可恢复的数据库配置/存储错误使 benchmark 进入
配置阻塞，不能用定时 retry 代替人工修复。`invalid_structured_output`、
`incomplete_output`、`context_overflow` 或 `policy_rejected` 只使对应 episode、维护任务或
QA item 成为终态失败，不暂停其他 spaces，也不永久阻塞当前 space 的后续来源顺序。

第一版先完整实现 extraction、Subject linking、Subject review、Subject split 和 search，
再执行正式实验；ablation study 留到以后。实验按配置直接报告各项结果，不选择优胜配置，
也不要求把模型因素与系统设计因素隔离。

每个 benchmark question 使用原始问题文本直接执行一次 public search，不调用 LLM 改写
query，也不执行迭代检索。除数据集官方 QA 指标外，实验只额外报告：

- 记忆写入与整理过程的 LLM 调用总量；
- 上述调用的 token 总量，只给出一个合计值，不拆分 prompt、completion、input、output
或 generation provider 使用的其他子类别；benchmark QA 回答和评测调用不计入该写入整理总量；
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

#### 1.2.7 实验版记忆构建日志

每次 memory-space build 至少生成两份按处理顺序追加的日志：一份 JSONL 结构化事件日志，
用于机器分析、统计和定位失败；一份 Markdown 高可读性审计日志，用于人工完整复盘 memories
和 subjects 如何形成及演化。两份日志共享 build/run ID、memory-space ID、episode ID、
source sequence、domain operation ID 及正式对象 ID，使同一事件可以相互对应。

这两份日志是实验产物，不是 SQLite 正式数据或可写事实源，不能反向驱动记忆状态，也不能
进入后续 LLM 输入。高可读性日志包含完整 benchmark 对话和 memory 内容，必须按包含原始
对话数据的敏感实验产物保存。

##### 结构化事件日志

结构化日志记录关键事件，不承担完整正文快照。每条事件至少包含时间、severity、事件类型、
关联 IDs、状态或结果，以及适用的对象数量、耗时和简短原因。覆盖范围至少包括：

- memory-space build 的开始、完成、暂停、恢复和失败；
- episode 开始处理、extraction 完成、Subject linking 完成和原子提交；
- 一次 episode 提取、创建和写入的 memory 数，新建 subject 数及新建 link 数；
- 记忆写入与整理阶段每次 LLM 调用的阶段、attempt、结果、耗时和单一 `total_tokens`；不记录
prompt/completion/input/output 等 token 子类别；
- Subject review 和 Subject split 的触发、开始、完成或失败，以及涉及的 subject ID；
- split 的 `full_split`、`partial_split` 或 `defer_split` 结果；
- review 导致的 memory 更新数、全局退役数和 summary 更新；
- model provider `error_class`、structured-output retry、warning、终态单项失败、临时暂停、
配置阻塞和恢复；
- partial split 移出的 memory 数、新建 subject 数，以及 full split 退役的原 subject。

结构化日志保留正式 IDs 和计数，便于汇总实验指标；不为了日志给 memory 增加 name 字段，
也不把 prompt/completion/input/output token 等子类别扩展成实验报告指标。

##### 高可读性审计日志

高可读性日志按 episode 来源顺序和后续维护实际发生顺序展开。它必须展示已经通过校验并
参与正式状态决策的完整内容，而不只是计数或对象 ID。

每个 episode 在 extraction 和 Subject linking 完成后展示：

- extractor 实际读取的规范化 episode 内容，包括 message 顺序、speaker、role、已知来源
时间和正文；不包含 dataset question、answer、evidence 等评测监督字段；
- extraction 得到的全部 memory contents，以及每条 memory 最终 link 到的 subject names；
如果结果是 `no_valuable_memory`，明确展示该语义结果；
- 本批新建 subjects 的 name 和初始 summary；
- 本批原子提交的最终结果。

memory unit 没有正式 name。日志为同一段落内的 memories 分配 `M1`、`M2` 等仅供阅读的
短标签，并同时展示完整 content；短标签不能保存为领域字段或跨操作身份，跨段落引用使用
正式 memory ID。

每次 Subject review 展示：

- review 前的 subject name、summary，以及全部 active memories 和 link basis；
- provenance request 和系统返回的来源 episodes（如果发生）；
- 每条被修改 memory 的正式 ID、修改前 content、修改后 content，以及 provenance 的前后
完整集合；
- 每条全局退役 memory 的 content，以及因此关闭的全部 active subject links；
- 未改变的 memories 可以按 ID 和 content 列出一次，无需伪造 change；
- review 后的完整 subject summary 和最终 active memory 集合。

每次 Subject split 展示：

- split 前原 subject 的 name、summary，以及全部 active memories 和 link basis；
- `full_split`、`partial_split` 或 `defer_split` 的结果和理由；
- full split 后全部新 subjects 的 name、summary、memory membership 和 link basis；
- partial split 新建的 subjects、移出的 memories、继续留在原 subject 的 memories，以及
原 subject 的新 summary；
- defer split 的 warning 和后续仍会在新增 link 后重试的说明。

高可读性日志还按实际发生位置展示 model provider 错误类别、structured-output 修复重试、
warning、error、终态单项失败、临时暂停、配置阻塞和恢复，使一次 memory-space build 可以
仅凭该日志按时间顺序复盘。

### 1.3 记忆检索



#### 1.3.1 公开 `search`

`search(query)` 只读取 active memories、active subjects 和 active links。第一版所有通道
仅执行精确向量检索，不使用 BM25、关键词、融合、重排或第二轮筛选。query 不设应用层字符
上限，也不静默裁剪。


| 配置项                                    | 值    | 含义                                                |
| -------------------------------------- | ---- | ------------------------------------------------- |
| `search_subject_top_k`                 | 5    | 公开 search 的 Subject 通道最多召回的 subject 数             |
| `search_subject_min_cosine_similarity` | 0.25 | 公开 search 的 Subject 通道最低 query-subject name 余弦相似度 |
| `search_subject_attached_memory_k`     | 1    | 每个公开 search Subject 通道结果附带的关联 memory 数            |
| `search_memory_top_k`                  | 15   | 公开 search 的 Memory 通道最多召回的 memory 数               |
| `search_memory_min_cosine_similarity`  | 0.35 | 公开 search 的 Memory 通道最低 query-memory 余弦相似度        |
| `search_memory_attached_subject_k`     | 1    | 每个公开 search Memory 通道结果附带的关联 subject 数            |


Subject 通道使用 subject name embedding，召回最多 5 个相似度不低于 0.25 的 subjects；
每个 subject 附带 active linked memories 中与 query 最相似的 1 条。Memory 通道召回最多
15 条相似度不低于 0.35 的 active memories；每条 memory 附带 active linked subjects 中
name embedding 与 query 最相似的 1 个。

通过各通道 top-k 和阈值的所有对象都进入最终结果。合并后只按照真实 links 去重并整理
subject-memory 对应关系，不计算融合分数、不重新排序、不再次淘汰。direct/contextual links
均参与，第一版不调整权重。由通道数量可派生出去重前最多 20 个 subjects 和 20 条 memories，
不把该结果重复配置为另一个上限。

不设置最终 subject 数、memory 数、每 subject memory 数、返回文本总字符数、summary 返回
字符数、单条 memory 返回字符数或超限最小保留数量。Public library 返回结构化结果；面向
主 Agent 的文本仅是该结果的确定性呈现。

#### 1.3.2 NumPy 精确扫描

实验版不使用 query embedding cache。检索从 SQLite 分批读取当前 memory space、active
retrieval signature 和目标实体下的全部向量，每批最多 8,192 行。归一化 query 向量与
目标向量矩阵使用 NumPy 矩阵乘法计算 cosine similarity，并持续维护全局 top-k；分批只
限制内存，不改变所有向量参与计算的精确语义。相同浮点值以稳定 ID 决定顺序，不设置浮点
近似并列容差。

### 1.4 LLM 结构化输出参考草图

本节记录四个记忆构建阶段当前使用的精确结构化输出契约。它们不是 public API，但字段名、
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

#### 1.4.2 Batch Subject linking

最终 linking 结果一次覆盖本 episode 组织批次中的全部 memories。新 subject 先使用批次内
`subject_ref`，每条 link 的目标通过判别字段引用已有 subject ID 或本批新 subject：

```json
{
  "result": "links",
  "new_subjects": [
    {
      "subject_ref": "new_subject_1",
      "name": "Mike's dietary preferences",
      "summary": "Mike's established dietary preferences and related constraints."
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
      "basis": "contextual"
    },
    {
      "memory_ref": "memory_2",
      "subject": {"kind": "new", "subject_ref": "new_subject_1"},
      "basis": "direct"
    }
  ]
}
```

如果 linking 阶段使用一次可选的主动关联检索，中间请求可以采用以下形状；获得结果后仍需
返回覆盖整个批次的最终 linking 结果：

```json
{"result": "association_search", "query": "possible relationship or impact direction"}
```

程序校验所有临时引用、已有 IDs、每条 memory 的 direct link 和 link 数量约束，并在应用前
验证已有 IDs 属于提供给模型的合法候选集合。

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
  "retirements": ["memory-uuid-3"],
  "summary": "Complete summary of the subject after applying the review."
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
      "summary": "...",
      "links": [
        {"memory_id": "memory-uuid-1", "basis": "direct"},
        {"memory_id": "memory-uuid-2", "basis": "direct"}
      ]
    },
    {
      "subject_ref": "new_subject_2",
      "name": "Mike's travel plans",
      "summary": "...",
      "links": [
        {"memory_id": "memory-uuid-2", "basis": "contextual"},
        {"memory_id": "memory-uuid-3", "basis": "direct"}
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
  "remaining_summary": "Summary supported only by memories remaining in the original subject.",
  "new_subjects": [
    {
      "subject_ref": "new_subject_1",
      "name": "Mike's dietary preferences",
      "summary": "...",
      "links": [
        {"memory_id": "memory-uuid-1", "basis": "direct"},
        {"memory_id": "memory-uuid-2", "basis": "direct"}
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

`defer_split` 是语义决定；full/partial split 中的非法 ID、不完整覆盖、数量越界或缺少
direct link 仍是需要修复的输出错误。

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
内顺序。已经 sealed 的 episode 和 blocks 不允许改写或重新切分。提取 overlap 直接读取
前一个 episode 尾部的 message；overlap 只作为上下文，不改变 provenance。

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
2. **Seal episode**：写入 episode 和 episode blocks，删除对应 pending blocks 和临时
  embeddings，并建立待提取状态。
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
`add_into_buffer`。正常 Agent 集成由宿主生命周期 Hook 自动调用；第一版不向主 Agent LLM
暴露主动 add 工具，避免同一交互经 Hook 与 tool 重复写入。

每次 connector add 至少包含一轮 user message 到 assistant final message 的完整交互。
connector 只投影 user/assistant messages；tool call、arguments 和 result 在读取 transcript
时识别并跳过。Assistant 即使引用了 tool result，Memory Engine 也只使用 assistant message
中明确表达的正文，不能回读 tool result 作为证据。

Receive transaction 成功、message 可靠进入 durable buffer 后即可返回，不等待 boundary、
extraction、linking、review 或 split。幂等 key 与 canonical preprocessed payload hash 相同的
重放返回原 receipt；相同 key 不同 hash 返回 conflict。调用方负责按来源顺序提交同一 stream，
core 使用 `ingestion_version` CAS 防止覆盖；不同 streams 可以并行接收。

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

#### 2.2.6 单 worker 与失败恢复

正式版只有一个顺序后台 worker，依次执行 boundary detection、extraction、link/write 和
Subject maintenance；review 与 split 的触发和同时满足时的先后顺序沿用 1.2.4--1.2.5。
该顺序是架构不变量，不配置 worker 并发、claim batch、
poll interval、lease、heartbeat、任务 timeout、任务级 retry/backoff、失联重领、恢复扫描
batch、优雅退出时间或 backlog 告警。实验版 benchmark 的并发参数不改变正式版模型。


| 配置项                                     | 值     | 含义                                                        |
| --------------------------------------- | ----- | --------------------------------------------------------- |
| `memory_pipeline_failure_pause_seconds` | 300 秒 | 临时依赖错误在操作内重试耗尽且 provider 没有给出有效恢复时间时，写入流水线在下一次恢复探测前至少暂停多久 |


Generation 和 embedding provider 错误统一按 1.1.12 的 `error_class` 处理。
`transient_transport`、`rate_limited` 和 `service_unavailable` 在操作内 transport retry 耗尽
后进入临时暂停；带有明确恢复时间的 `quota_exhausted` 直接进入临时暂停。SQLite 可重试
transaction/连接错误在 transaction retry 耗尽后执行相同策略。`rate_limited` 或
`quota_exhausted` 带有明确恢复时间时将 `pause_until` 设为该时间，其他临时错误设为当前时间
加 300 秒。暂停期间不执行 boundary、extraction、link/write、review 或 split，search 仍
读取已提交记忆。

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
输入错误直接成为终态单项失败。以上单项失败都不触发全局暂停，也不能反向拒绝已经成功
durable add 的交互。

正式版使用按需启动、队列排空后退出的本地 worker，不依赖 Hook、MCP、CLI 或 TUI 中任一
进程长期存活。Hook 完成 durable add、创建 flush request 或其他入口发现 pending 工作时，
只负责确保 worker 已被唤醒或启动，不等待记忆构建。多个进程通过本地跨进程互斥保证同一
时间只有一个顺序 worker；互斥与进程启动的具体实现由落地时确定，不引入 lease、heartbeat
或常驻 daemon。

启动 worker 失败或 worker 中途退出不会删除 pending 状态；下次 Hook、CLI、TUI、MCP 或
FluxFold 启动活动再次唤醒。进程崩溃后同样根据 SQLite pending state 继续，进程内 queue、
buffer view 和 LLM 上下文都不是事实源。

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
`session_id`、`transcript_path` 等 payload 定位本轮新增交互。resume 同一 session 复用原
stream；不同 sessions 使用独立 buffers，但同一 user 的 streams 共享默认 memory space。
Hook 可以是短生命周期子进程，search 通过由 Claude Code 管理的本地 stdio MCP server
暴露；两者共享 SQLite，不要求共享进程内存，也不构成 FluxFold 独立 daemon/service。

宿主能够报告 session 结束时，connector 必须先读取并 durable add 尚未接收的全部完整
transcript records，再对该 stream 创建覆盖最后一个已接收 block 的 flush request，使开放
buffer 的完整尾部立即进入处理。`/exit`、正常关闭等可观测退出采用此路径；强制杀进程、
宿主崩溃或断电没有 Hook 执行机会，不承诺即时 flush，但下次 FluxFold 活动必须根据 cursor、
durable inbox 和 pending flush 状态恢复并处理遗留尾部。未写完的 transcript record 不能作为
完整输入交给 extractor。

resume 继续原 stream；fork、复制或宿主产生新的 session identity 时建立新 stream，仍写入
同一 user 的默认 memory space。transcript 被截断、替换或 compaction 后，connector 不修改
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
文件由宿主环境轮转。实验版 benchmark 的双日志仍按 1.2.7 独立生成，不改变正式版默认日志
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
