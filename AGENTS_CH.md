# AGENTS.md（中文版）

> 本文件是 `AGENTS.md` 的中文对照版，两者必须同步修改并保持等价。

> **适用范围：仅实验版。** 本文件约束 FluxFold **实验版**（面向 benchmark 的 Memory Engine）
> 的实现。第一版正式版启动前会重写本文件。不要把它的规则外推到 CLI、TUI、connector 或任何
> 正式版议题。

## 项目概述

FluxFold 是从零构建的 agent memory system。核心判断：**把全部 LLM 成本前移到写入路径，读取
路径保持纯向量检索。** 记忆组织在 `subject` 中——一种规模有界、会自我具体化的组织单元，随着
增长被审核（review）和分裂（split）。

- 交付形态：Python **library**。不提供 HTTP/gRPC service、daemon 或网络监听端口。
- distribution / import package / CLI 名称统一为 `fluxfold`，源码位于 `src/fluxfold/`。
- Python `>=3.12`，使用 `uv` 管理，单 distribution，`src` layout，Apache-2.0。

## 权威文档


| 文档                            | 作用                                       |
| ----------------------------- | ---------------------------------------- |
| `design_detailed.md` §1.1–1.4 | 实验版的**实现事实源**：数据模型、存储、配置数值、处理流程、结构化输出形状。 |
| `design.md`                   | 项目级交付形态、工具链、质量基线、演进条件。                   |
| `background.md`               | 每个机制存在的原因，以及哪些假设尚未验证。                    |


`design_detailed.md` §2.x 属于**正式版**设计，不在范围内，不要实现。
文档与代码冲突时以文档为准；如果设计确实有误，在同一次提交中修改文档，绝不静默偏离。

## 实验版范围

**范围内：** memory extraction、候选召回、批次 Subject linking、Subject review、Subject split、
公开 `search`、SQLite 持久化、embedding 管理、`LongMemEval-S` 与 `LoCoMo_refined` 的
dataset adapter、benchmark runner、构建日志、测试。

**范围外（不要实现）：** CLI、TUI、connector、MCP server、durable inbox、buffer、基于 embedding
的 situation boundary detection、`flush`、users/streams 表、ablation study、图数据库、向量数据库、
ANN 索引、BM25、reranker、query 改写。

## 目标布局

```text
src/fluxfold/           # library core；不得 import benchmark 或 adapter 代码
tests/                  # pytest，目录结构对应 src/fluxfold/
benchmarks/             # dataset adapters + runner；依赖 library，不可反向依赖
scripts/setup-dev.sh    # 一次性本地环境 bootstrap
pyproject.toml uv.lock .python-version LICENSE README.md .env.example
design.md design_detailed.md background.md AGENTS.md AGENTS_CH.md
```

依赖方向单向：`benchmarks/` → `src/fluxfold/`。Dataset adapter 只把来源记录规范化为 episode，
不得包含 extraction、linking、review、split、写入或 search 逻辑。

## 命令

```bash
./scripts/setup-dev.sh           # 首次环境 bootstrap
uv sync                          # 刷新环境
uv run ruff format --check .     # 格式
uv run ruff check .              # lint
uv run mypy src/fluxfold         # 类型
uv run pytest                    # 测试
uv build                         # wheel + sdist
```

工具一律通过 `uv run` 执行。四项检查全部通过后，改动才算完成。

## 核心流程

```text
dataset session
→ 规范化并持久化不可变 episode（content_hash、source_sequence）
→ memory extraction                 （0..N 条自包含 memory，或 no_valuable_memory）
→ 逐条 memory 的候选召回            （Subject 通道 + Memory 通道，可选 1 次 association_search）
→ 批次 Subject linking              （一次 LLM 输出覆盖整个 episode 批次）
→ 原子提交                          （memory、version、provenance、embedding、subject、link、completion）
→ Subject split / Subject review    （两者同时满足时先 split）
```



## 编码规则

- Fail fast。只捕获真正能恢复的异常；意外异常直接崩溃暴露。
- 不为不可能发生的错误写防御分支或兜底（包括数据模型已经排除的情况）。只在边界（dataset 输入、
model provider 输出）做校验。
- 实验版改动不做兼容性设计：不为“以前的草案用过”而保留不再使用的 API、schema 字段、配置项、
migration、shim 或 feature flag。
- 修改时删除所有不会再使用的死代码，保持代码简洁；不留注释掉的残留、无用 import 或死 helper。
- 代码中的注释、docstring、LLM prompt 以及其他 in-code 文本一律使用英文。
- 不做投机性抽象。三行相似代码好过过早抽象。
- `src/fluxfold/` 的 public API 必须完整类型标注并通过 `mypy`。
- Ruff 同时承担 formatter 与 linter，target 为 `py312`。



## 文档义务

- 行为变化必须在同一次改动中写入 `design_detailed.md`（Memory Engine）或 `design.md`（项目级）。  
没有配套文档更新的代码改动是不完整的。
- 只描述最终的当前设计。删除被取代的描述，而不是记录“某机制已不再使用”；历史由 git 保存。
- `AGENTS.md` 与 `AGENTS_CH.md` 必须同步更新并保持等价。

