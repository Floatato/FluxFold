# FluxFold Design

本文档记录 FluxFold 当前商定的项目级设计、实现边界和演进条件。正文中的方案均为当前有效设计；尚未进入当前实现范围的实验和优化统一记录在“简化方案的演进原则”章节。

## 交付形态

FluxFold 的第一版正式版作为一个可嵌入其他应用的 library 开发，并同时提供一个带有初始 TUI 的 CLI。

- library 是首要交付物，也是核心能力的唯一实现位置。
- CLI 和 TUI 是 library 的薄适配层，负责接收用户输入、调用 library API，并向用户呈现结果。
- “薄适配层”约束的是业务逻辑归属，而不是界面的复杂度；TUI 可以逐步增加页面和交互，但不得在界面层重复实现核心能力。
- CLI 和 TUI 在正式版阶段随 library 功能逐步完善；第一版正式版只实现开发和基本使用所需的入口与交互。
- CLI 不应复制 library 中的业务逻辑，也不应形成一套独立的行为语义。
- 当前不提供需要独立部署和管理的常驻 service，不设计 HTTP、gRPC 等远程 API，也暂不承担服务部署、认证、租户隔离、限流和远程并发管理等职责。
- 是否增加 service 留待出现明确的跨语言调用、远程访问、多客户端共享或集中部署需求后重新评估。

### 影响

- 对外 API 的设计首先以 library 使用者为中心。
- 项目结构需要让 CLI 依赖 library，而不能让 library 依赖 CLI。
- 核心能力应与终端输入输出解耦，以便 library 可以直接在其他程序和测试中使用。
- 实验版工程只需覆盖 Memory Engine library、数据集 adapters、benchmark runner 和测试；第一版正式版再加入 CLI entry point、TUI 与 connector，两个阶段都不需要引入 Web 服务框架和服务部署基础设施。



## 分阶段落地范围

FluxFold 在第一版正式版之前先实现一个面向 benchmark 的实验版，在实验范围内完整实现
Memory Engine 的核心流程后再开始正式实验。

### 实验版

- 实验版重点实现记忆添加、整理和检索，包括 memory extraction、批次 Subject linking、
  Subject review、Subject split 和公开 `search` 的完整流程。详细流程见
  `design_detailed.md` 1.2～1.3 节。
- 实验版提供能够每次加入一个数据集 session 的 `add` API。数据集 session 作为已经
  划定边界的输入片段直接进入记忆提取流程。详细入口与规范化协议见
  `design_detailed.md` 1.1.13 和 1.2.1 节。
- 实验版提供满足 `LongMemEval-S` 与 `LoCoMo_refined` 评测要求的 `search` API，并为
  两个数据集实现必要的输入适配、运行脚本和结果输出。数据集映射与检索规则分别见
  `design_detailed.md` 1.2.1 和 1.3 节。
- 实验版以能够接入两个数据集完成端到端跑分并输出既定实验指标为完成条件。实验直接报告
  各项结果，不选择优胜配置；ablation study 留到以后。执行与报告规则见
  `design_detailed.md` 1.2.6 节。
- 实验版不实现 CLI、TUI、基于 embedding 的 situation boundary detection、Claude Code
  connector 或其他宿主接入。

### 第一版正式版

- 第一版正式版在实验版 Memory Engine 核心能力之上，补齐真实 Agent 交互的自动采集、
  durable inbox、buffer、基于 embedding 的 situation boundary detection 和
  situational episode 封装。详细流程见 `design_detailed.md` 2.1～2.2 节。
- 第一版正式版实现 CLI、TUI 和首个 Claude Code connector，并按照正式使用场景补齐
  lifecycle、幂等、并发、恢复和工程质量要求。
- 实验版与第一版正式版共用同一个 Memory Engine library core；benchmark adapter、
  connector 和 CLI/TUI 只负责各自的输入输出适配，不复制记忆业务逻辑。



## Memory Engine 详细设计

Memory Engine 的当前行为、数据表示、全部已定配置和处理流程统一记录在
`design_detailed.md`：实验版见 1.1～1.4 节，第一版正式版补充设计见 2.1～2.4 节。本文件
只保留项目级概述和演进条件，不重复实现细节。

## 主 Agent 接入架构

FluxFold 通过宿主专用 connector 接入可替换的主 Agent。connector 是位于宿主与 FluxFold library API 之间的外围适配层，不是主 Agent 本身；它负责把宿主生命周期事件和消息格式转换为 FluxFold 的统一交互语义。Memory Engine core 不依赖 Claude Code、Codex 或其他特定宿主。

- `add` 与 `search` 是 FluxFold library 对调用方提供的核心 operation。实验版 `add` 接收已经规范化的 dataset episode 并直接进入记忆构建；正式版 `add` 接收宿主无关的结构化交互并委托给内部 `add_into_buffer`。两阶段复用同一套 episode-to-memory 核心逻辑，但入口语义分别遵循 `design_detailed.md` 1.1.13、1.2.1 和 2.2.1 节。`search` 返回可独立渲染的结构化事实结果，详细契约见 1.1.13 和 1.3 节。
- library 另提供幂等的 `flush(stream_ref, idempotency_key)` lifecycle control，用于要求系统处理某个 stream 当前 buffer 的尾部。`flush` 不写入新的交互或记忆，不把 stream 永久关闭，因此不作为第三种 memory semantic operation；详细语义见 `design_detailed.md` 2.1.6 和 2.2.7 节。
- `add_into_buffer` 是正式版 library 内部的 buffer 写入操作，不依赖具体触发方式。正常 Agent 集成由宿主生命周期 Hook 通过 connector 自动调用正式版公共 `add`；实验版 benchmark adapter 则向实验版入口提交规范化 dataset episode。主 Agent 的 LLM 不负责决定自动采集是否发生，也不决定从本轮交互中挑选什么内容写入。
- `search` 是主动检索通道，以 MCP 或宿主原生工具暴露，由主 Agent 的 LLM 决定何时调用以及 query 内容；正式版接入边界见 `design_detailed.md` 2.3 节。
- 自动写入通道和主动检索通道职责分离；第一版不向主 Agent 暴露可由模型主动调用的 `add` 或 `add_into_buffer` 工具，避免同一交互经 Hook 和工具重复写入。这里不限制 benchmark、connector 或普通应用代码调用各自阶段的 public library 入口。
- `add_into_buffer` 在原始交互经过确定性预处理并可靠持久化到 durable inbox 后即可向公共 `add` 的调用方确认接收，不等待基于 embedding 的 situation boundary 检测、situational episode 生成或 `memory unit` 与 `subject` 整理完成。connector 收到成功结果后再结束对应 Hook；详细事务边界见 `design_detailed.md` 2.1.7 和 2.2.1 节。
- connector 必须支持重复触发或重试下的幂等接收。正式版 receipt、payload hash 与冲突语义见 `design_detailed.md` 2.1.3 和 2.2.1 节；稳定来源身份和 transcript 连续性规则见 2.2.8 节。
- 正式版不同 streams 可以并行接收；同一 stream 的新 interactions 必须由调用方按来源顺序提交，core 使用 stream version/CAS 防止并发覆盖。FluxFold 不根据调用到达时间猜测来源顺序；详细并发语义和按需单 worker 规则见 `design_detailed.md` 2.1.7、2.2.1 和 2.2.6 节。
- 外部模型服务统称 model provider，其中生成记忆结构化结果的是 generation provider，生成
  retrieval 或 boundary vector 的是 embedding provider。provider adapter 将厂商错误归一化
  为公共 `error_class`：临时传输、限流和服务不可用执行有限重试，耗尽后临时暂停；认证、
  配置、硬配额和非法请求进入配置阻塞；context overflow、policy rejection 以及修复耗尽的
  结构化输出错误只终止当前 item。错误分类、重试参数和恢复语义见 `design_detailed.md`
  1.1.12 和 2.2.6 节。



### 首个 Claude Code connector

- 第一版正式版首先实现 Claude Code connector。
- Claude Code 的 `Stop` Hook 在每次正常的 Assistant 最终回复结束后触发 connector。Hook
  payload 提供 `session_id`、`transcript_path` 等信息；connector 从
  transcript 定位本轮完整交互，只按来源顺序提取其中的 user/assistant messages，跳过 tool
  call、tool arguments 和 tool result，再转换为统一 schema 并调用公共 `add`。
- Claude Code 能够通知的 session 结束事件触发尾部提交与 `flush`：connector 先 durable add
  尚未接收的完整 transcript records，再以最后一个 block 为 watermark 请求处理当前 stream
  的全部完整尾部。强制终止、宿主崩溃或断电无法保证即时 Hook，遗留 durable 状态在下次
  FluxFold 活动时恢复处理。
- 每个 Claude Code session 使用独立 `stream_id` 和 buffer；resume 同一个 session 时继续使用
  原 `stream_id`，不同 session 的 messages 不进入同一个 situational episode。
- 第一版正式版按 user 建立默认 memory space；同一 user 的多个 streams 可以共享记忆组织和检索范围，但不能混用 buffer。
- FluxFold 通过由 Claude Code 启动的本地 stdio MCP server 暴露 `search`，由 Claude Code 的 LLM 主动调用。
- Hook 可以作为短生命周期子进程运行，stdio MCP server 可以作为由宿主管理的会话期子进程运行；二者使用同一个持久化存储，不要求共享 Python 进程内存。
- 这种宿主管理的本地进程不视为 FluxFold 提供独立常驻 service；当前仍不引入 daemon、网络监听端口或需要单独部署的服务。
- 每次成功接收或显式 flush 都确保按需本地 worker 已被唤醒；多个入口通过跨进程互斥只允许
  一个顺序 worker 推进 durable pending 状态，队列排空后 worker 退出。
- Claude Code connector 随主 package 发布，由 FluxFold CLI 显式安装、检查、更新和卸载。
  安装只合并 FluxFold 自己的 Hook、MCP 和使用说明，修改前建立可恢复备份；卸载只移除
  FluxFold 能确认拥有的配置，不删除数据库和 memories。
- 后续目标至少包括 Codex、Pi、OpenClaw 和 Hermes Agent；新增宿主时实现新的 connector，不修改 Memory Engine 的核心语义。

transcript 投影、退出恢复、配置所有权和本地进程边界见 `design_detailed.md` 2.2.6～2.2.8、
2.3 和 2.4.1 节。



### 影响

- connector 属于 Memory Engine 外围适配层，可以依赖宿主协议；core 不得依赖宿主配置文件、Hook payload 或 MCP 类型。
- TUI 中使用 `/connectors` 查看和管理宿主集成，而不是用 `/agents` 表示实时在线 agent。
- connector 状态应描述是否安装、是否配置、最近一次写入或错误等可验证事实；没有实时通信时不得声称某个主 Agent 当前在线。
- connector 无法安全识别 transcript 增量、配置格式或已安装配置的所有权时必须记录明确
  warning 或停止修改，不能靠猜测覆盖用户状态。
- 当前“无 service”决定允许宿主启动 Hook/MCP 子进程，但不允许在没有新决策的情况下演变为常驻远程服务。



## CLI 与 TUI 交互形态

FluxFold 使用 `Typer` 实现外层 CLI，使用 `Textual` 实现全屏 TUI，并在 TUI 内维护独立的 slash command registry。

- 安装后的 `fluxfold` 命令是统一入口；在交互式终端中不带子命令运行时，默认启动 TUI。
- `Typer` 负责 TUI 启动入口以及 `--help`、版本查询和未来面向 shell 或自动化脚本的一次性命令。
- `Textual` 负责终端界面、输入组件、页面或弹窗、键盘事件，以及耗时操作期间的非阻塞交互。
- `/help`、`/provider`、`/connectors`、`/exit` 等 slash commands 只在 TUI 内生效，不作为 `Typer` subcommands 解析；`/provider` 统一管理 generation 与 embedding provider 配置。
- slash commands 通过集中 registry 注册和分发；command handler 只负责把交互意图转换为 application/library 调用及可展示的结果。
- generation/embedding provider 配置、connector 查询和 memory 操作等实际能力必须由 TUI 之外的 application/library API 提供，以便嵌入式调用和测试不依赖终端界面。
- CLI 启动 TUI 时，TUI 与 library 可以在同一个 Python 进程内运行；主 Agent connector 可以按上一节采用宿主管理的 Hook/MCP 子进程。
- 第一版 CLI 提供 generation/embedding provider 与非秘密配置、connector 生命周期与诊断、
  search、状态检查、`flush`、数据只读检查、删除指定或全部 memory spaces、数据库备份和
  `doctor`；`doctor` 可以执行 model provider health check 和显式恢复探测。TUI 覆盖
  其中的高频交互。人工修改单条 memory/subject、数据导入和在 TUI 中运行 benchmark 不在
  第一版范围。详细边界见 `design_detailed.md` 2.4.2 节。



### 影响

- `Typer` 和 `Textual` 作为当前单一 distribution 的运行时依赖，不额外拆分 CLI/TUI package 或 optional extra。
- 项目结构需要为 CLI entry point、Textual app 和 slash command registry 保留清晰边界，同时避免核心模块依赖任何终端 UI 类型。
- TUI 中的耗时或异步操作不得阻塞界面事件循环，应通过 Textual 提供的后台任务机制调用 application/library API。
- 如果未来需要远程访问、集中部署或由 FluxFold 自主管理常驻进程，必须重新评估当前“不提供独立常驻 service”的决定。



## 基础技术栈与项目布局

FluxFold 采用 Python 3.12 及以上版本、`uv`、单 package 和 `src` layout。

- `pyproject.toml` 中的 `requires-python` 设为 `>=3.12`。
- 使用 `uv` 管理 Python 版本、项目虚拟环境、依赖解析、依赖安装和 lockfile。
- `uv.lock` 应提交到版本控制，以复现本地开发和 CI 环境。
- FluxFold 作为单个 Python distribution 开发；现阶段不拆分 core、CLI 或其他子 package，也不建立 workspace。
- 可导入 package 使用 `src/fluxfold/` 目录，公共 import 名称为 `fluxfold`。
- 实验版运行时依赖包括 NumPy，用于 SQLite 之上的精确向量扫描；不暴露 CLI entry point，也不加入 Typer 或 Textual。
- 第一版正式版仍使用同一个 distribution：通过 `pyproject.toml` 的 `[project.scripts]` 暴露 `fluxfold` 命令，并加入 Typer 与 Textual。
- 首次建立本地开发环境时运行 `scripts/setup-dev.sh`：安装 `.python-version` 指定的 Python、按 `uv.lock` 同步环境，并执行与日常开发相同的质量检查和 `uv build`。该脚本只负责一次性 bootstrap，不是任务运行器；日常开发仍直接使用 `uv run`。



### 影响

- 测试和开发命令应在安装后的 package 语义下运行，避免依赖仓库根目录恰好位于 `sys.path` 的行为。
- CLI 模块可以依赖其他 `fluxfold` 模块，核心 library 模块不得反向依赖 CLI。
- 工程工具应优先通过 `uv run` 执行；CI 应以 `uv.lock` 为依赖解析依据。
- 只有出现独立发布、独立版本或明显不同的依赖边界时，才重新评估是否拆分多个 package。



## 项目标识、版本与 repository 范围

### 名称

- 项目展示名称为 `FluxFold`。
- Python distribution name 为 `fluxfold`，用于 `pyproject.toml`、PyPI 和 `uv add fluxfold` 等安装场景。
- Python import package name 为 `fluxfold`，源码位于 `src/fluxfold/`，使用方式为 `import fluxfold`。
- CLI command name 为 `fluxfold`。
- 初期保持这几个名称一致，降低安装名、导入名和命令名不同带来的认知成本。
- 截至 2026-08-15，PyPI 的 `fluxfold` 项目页面返回 404，可将其作为当前 distribution name；这不构成名称保留，首次发布前必须再次核验可用性和潜在商标冲突。
- 如果发布前 `fluxfold` 已被占用或存在法律/品牌冲突，优先只调整 distribution name，例如改为 `fluxfold-memory`，同时尽量保留 import package 和 CLI command 为 `fluxfold`。



### 初始版本

- `pyproject.toml` 的初始版本设为 `0.1.0`。
- `0.x` 阶段表示 FluxFold 正在完成架构验证，public API、持久化格式和 connector 行为仍可能发生不兼容变化。
- 每次发布仍应提供明确的变更说明；“版本小于 1.0”不能替代数据迁移说明或对使用者的破坏性变更告知。
- 当核心 `add`/`search` 语义、持久化兼容策略、至少一个 connector 和基本发布流程经过真实使用验证后，再评估 `1.0.0`。



### License

- FluxFold 采用 Apache License 2.0，SPDX identifier 为 `Apache-2.0`。
- repository 根目录在初始化时加入完整的 `LICENSE` 文本，`pyproject.toml` 使用对应 SPDX license expression。
- 选择 Apache-2.0 是因为它是宽松开源许可，同时包含明确的 copyright 和 patent license 条款，适合作为可嵌入其他 Agent 的基础 library。
- 初期没有额外 attribution 时不创建空的 `NOTICE`；如果未来引入要求保留 attribution 的内容，则随 distribution 维护准确的 `NOTICE`。
- license 选择会影响所有贡献和发布，不将换 license 视为常规工程升级；接受外部贡献前应再次确认版权主体和贡献政策。该决定不替代法律意见。



### README 当前定位

- repository 根目录使用英文 `README.md` 作为面对使用者和新贡献者的项目入口；架构讨论和决策细节继续以本文件为准。
- 初始 README 只包含：项目定位、当前 early-development 状态、核心能力概览、Python/uv prerequisites、开发环境建立方式、质量检查与构建命令，以及指向 `design.md` 的链接。
- 在 API 尚未实现前，不写无法运行的安装/使用示例，不宣称尚未通过 benchmark 验证的性能、准确率或兼容能力。
- 出现第一个可运行的 `add`/`search` vertical slice 后，再加入经过测试的 minimal usage；Claude Code connector 可用后，再加入安装和验证步骤。
- 当 README 因 API reference、connector 指南、配置说明或运维内容变得难以浏览时，将细节迁移到 `docs/`，README 只保留入口和最短路径。



### `.gitignore` 边界

- `.gitignore` 只排除可再生成的构建/缓存文件、本地环境、编辑器状态、coverage 产物和本地 secret files，不排除源码、测试、配置模板、`uv.lock` 或 migration files。
- 初始规则覆盖 `.venv/`、`__pycache__/`、`*.py[cod]`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.coverage`、`coverage.xml`、`htmlcov/`、`build/`、`dist/`、`*.egg-info/`、`.env`、`.env.*`、`.DS_Store`、`.idea/` 和 `.vscode/`。
- 允许提交不含真实 secret 的 `.env.example`；ignore 规则对 `.env.*` 使用 `!.env.example` 例外。`uv.lock` 和 `.python-version` 必须提交。
- FluxFold 的实际用户记忆和 model provider credentials 不得进入 repository。其本地存储路径确定后，再为该明确路径增加 ignore rule，不预先用宽泛规则隐藏可能应提交的 fixtures 或 examples。



## 构建方式

FluxFold 使用 `uv_build` 作为 Python build backend，并在初期保持为 pure Python package。

- `pyproject.toml` 的 `[build-system]` 使用 `uv_build`，由它将源码构建为标准 wheel 和 sdist。
- FluxFold 自身初期不包含需要编译的 C、C++、Rust 或 Cython extension，因此可以发布与操作系统和 CPU 架构无关的 pure Python wheel。
- “pure Python”只约束 FluxFold 自身的构建方式；FluxFold 仍可依赖包含 native code 的第三方 package。
- 不为尚未证实的性能需求预先引入 native toolchain、多平台 binary wheel 构建或相应发布流程。
- 如果性能测量表明第三方高性能依赖仍无法满足需求，再评估 Maturin、scikit-build-core 或独立 native acceleration package。



### 影响

- 初始构建流程只需验证 `uv build` 能生成可安装的 wheel 和 sdist。
- CI 不需要安装 Rust、C/C++ compiler 或处理平台相关 ABI。
- 引入 FluxFold 自有 native extension 属于需要重新评估 build backend、CI matrix 和发布流程的设计变更。



## 开发质量基线

FluxFold 初期使用 `pytest`、`pytest-cov`、Ruff 和 `mypy` 建立统一的测试、格式化、静态检查和类型检查基线。

- `pytest` 负责自动化测试，`pytest-cov` 负责收集测试覆盖率。
- Ruff 同时承担 formatter 和 linter，不再叠加 Black、isort 或 Flake8。
- `mypy` 负责静态类型检查；`src/fluxfold/` 中的 public library API 必须提供类型标注。
- Ruff 的 Python target version 与项目最低版本一致，设为 `py312`。
- `pytest`、`pytest-cov`、Ruff 和 `mypy` 放入 `pyproject.toml` 的 `[dependency-groups]` `dev` group，不进入发布给用户的 runtime dependencies。
- 本地开发和 CI 使用同一组 `uv run` 命令；CI 使用不会修改文件的检查模式。
- 首次建立本地开发环境时运行 `scripts/setup-dev.sh`，它会执行与下面相同的检查命令以及 `uv build`。日常开发不经过该脚本。
- 初期收集 coverage 数据，但不设置缺少实证依据的全局覆盖率硬门槛。
- 初期不引入 `tox`、`nox`、`pre-commit` 或额外任务运行器；只有重复操作或多环境管理产生明确成本时再增加。



### 初始检查命令

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/fluxfold
uv run pytest
```



### Memory Engine 测试数据集

- FluxFold 实验版使用 `LongMemEval-S` 和 `LoCoMo_refined` 作为 Memory Engine 的测试数据集。
- 两个数据集用于端到端评估长期记忆的添加、组织、维护和检索效果。完整实现实验版范围内的全部功能后再运行正式实验，并按 `design_detailed.md` 1.2.6 节记录配置、既定指标和结果，形成正式版实现前的对照基线。



### 影响

- 新代码在合并前应通过格式、lint、类型和测试检查。
- coverage 用于发现缺少测试的高风险路径，不把单一百分比当作测试质量本身。
- 工具配置集中保存在 `pyproject.toml`，避免本地开发和 CI 使用互相漂移的规则。



## 简化方案的演进原则

FluxFold 当前优先采用能够尽快形成端到端闭环的简单方案。是否扩展或替换方案，不以时间、代码总行数或抽象上的“将来可能需要”为依据，而以测试、benchmark、profiling、真实使用和维护成本暴露出的可观察压力为依据。

采用以下通用原则：

- **先验证问题，再增加机制。** 没有可复现的失败、性能数据、兼容性需求或维护成本时，不预先引入分布式系统、插件框架、native toolchain 或复杂任务编排。
- **优先扩展，再考虑替换。** 现有工具能够继续承担核心职责时，通过增加配置、测试层级或 adapter 扩展；只有职责不再匹配时才替换。
- **保持稳定边界。** 演进时优先保持 library API、持久化数据和 connector 语义兼容；无法兼容时需要迁移方案和明确的版本变化。
- **一次只解决已出现的压力。** 避免同时拆包、换存储、换 build backend 和引入 service，使问题来源和验证结果不可判断。
- **记录触发证据。** 重大演进应在本文档记录触发问题、采纳方案、替代方案、迁移影响和验证方式。



### 架构和交付形态


| 当前简单方案                                               | 需要重新评估的可观察条件                                                          | 大致演进方向                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 实验版仅交付 benchmark 所需的 library API 和数据集 adapters；第一版正式版采用 library-first 并附带 CLI/TUI | 出现远程访问、跨语言调用、集中部署、多客户端共享或统一权限管理需求 | 在保持 library core 的基础上增加独立 service/API adapter |
| CLI/TUI、library 位于一个 distribution                    | library 用户被迫安装明显不需要的 UI/connector 依赖，或组件需要独立版本、发布节奏和依赖约束              | 先使用 optional dependencies/extras；仍不足时拆为 core、CLI/connectors 等独立 distributions，并考虑 `uv` workspace |
| 单一 `src/fluxfold/` package                           | 模块边界长期互相反向依赖、不同部分需要独立所有权，或单元无法在不加载大量无关依赖的情况下测试                        | 先加强内部 package 边界；只有存在独立发布价值时再拆 distribution                                                      |
| Typer + Textual 的本地终端界面                              | 需要浏览器、多用户、远程管理、复杂可视化或终端能力无法表达的交互                                      | 保留 application/library 层，新增 Web/desktop presentation adapter；只有终端需求消失时才考虑替换 Textual              |
| 宿主管理的 Hook 与 stdio MCP 子进程，无 FluxFold daemon         | 后台任务因宿主退出长期积压、多个进程需要可靠协调、维护任务必须持续运行，或需要跨机器访问                          | 先引入可恢复的本地 worker/daemon；出现远程和多用户需求后再演进为正式 service                                                |
| 第一版正式版首先支持 Claude Code connector | 实现第二个宿主时出现重复接线逻辑或 Claude Code 特有概念泄漏到 core | 提炼稳定的 connector adapter contract 和兼容性测试；connector 依赖或发布明显分化后再拆插件 package |
| `fluxfold` 同时作为 distribution、import package 和 CLI 名称 | 发布前名称被占用、出现商标风险，或后续拆分多个 distributions                                 | 优先保持 import package 和 CLI 稳定，只调整 PyPI distribution 或为拆分包增加后缀                                     |
| 英文 README 作为单一项目入口                                   | API、connector、配置和运维文档使 README 难以快速找到安装与最小用法                           | 将详细内容迁移到版本化 `docs/`，README 保留定位、quick start 和导航                                                  |
| FluxFold 使用独立 Git repository                         | 多个 distributions 或 connectors 必须原子修改、统一测试和统一发布，且跨 repository 协调成为持续成本 | 在有明确共同发布边界时评估 monorepo/`uv` workspace；参考项目仍不纳入 FluxFold repository                               |




### Memory Engine


| 当前方案 | 需要重新评估或开展实验的条件 | 实验或演进方向 |
| --- | --- | --- |
| 每个 interaction stream 使用有界 SQLite durable ring buffer，单顺序 worker 从 SQLite pending state 恢复；model provider 临时错误、配置阻塞和终态单项失败按统一 `error_class` 区分。详细规则见 `design_detailed.md` 1.1.12、2.1.4、2.2.3 和 2.2.6 节 | 持续吞吐、宿主退出或多进程协调使单 worker 和固定容量淘汰无法满足真实使用，或错误分类在真实 provider 上无法稳定归一化 | 先调整 provider adapter、诊断和本地 worker coordination；达到多机规模后再评估消息队列 |
| 正式版仅在合法的 `assistant final → user` 候选点上比较 message embedding 差值，并按 soft、hard 与 flush 规则选择候选边界或整体 seal。详细规则见 `design_detailed.md` 2.2.4 节 | 带标注样本或端到端任务持续出现过切、欠切、延迟过高或成本不可接受 | 调整 embedding、阈值和 buffer 策略；必要时将 LLM situation boundary detection 作为对照方案 |
| memory extractor 按未来价值与独立生命周期提取 `0..N` 条自包含记忆，并保留 situational episode provenance。详细规则见 `design_detailed.md` 1.2.2 节 | 数据集或真实交互出现持续的提取遗漏、噪声、粒度不稳定、时间状态错误或来源归属错误 | 调整提取规则、结构化输出、字符数上限和 provenance 使用方式，并增加对应的定向评测 |
| 写入阶段的 Subject 通道仅使用 `subject name` 召回候选。详细规则见 `design_detailed.md` 1.1.7 和 1.2.3 节 | Subject 候选召回持续遗漏相关组织单元，或 summary 能够提供有效区分信息 | 对比仅使用 `subject name` 与使用 `subject name + subject summary` 的召回效果 |
| 写入阶段分别为新 memories 召回 Subject 与 Memory 候选，再由 LLM 对同一 episode 的整个组织批次一次性输出 links 和新 subjects；每条 memory 通常建立 `1～4` 个 link，并以 `direct` 或 `contextual` 记录依据。详细规则见 `design_detailed.md` 1.2.3 节 | 候选数量或相似度门槛造成漏召回、候选噪声，或两类 link 的判定不稳定 | 对召回参数和两个通道做消融，并分别评估漏链、错链与 `link_basis` 分类质量 |
| 写入阶段可配置一次主动 `association_search`。详细规则见 `design_detailed.md` 1.2.3 节 | 语义相似度较低但具有实质影响的关系持续漏链，或主动检索引入过多 link 和推理成本 | 对关闭与开启主动关联性检索做消融实验，并评估关联召回率、错误 link 和调用成本 |
| 新建 subject 时只建立当前 episode 组织批次中由 LLM 选定的 memory links，不回填更早 episode 的 memories。详细规则见 `design_detailed.md` 1.2.3 节 | 写入顺序导致早期记忆持续缺少后来才成立的 subject link | 评估有限回填本次候选中的旧 memory；采用前验证其是否造成零碎 subject、冗余 link 或 top-k 结果挤占 |
| subject 新增记忆达到阈值后，审核全部当前 active 关联记忆，并最多按需读取一次 provenance。详细规则见 `design_detailed.md` 1.2.4 节 | 去重、冲突处理、summary 质量、审核成本或一次来源读取无法满足要求 | 调整触发阈值、审核 schema、来源读取条件和允许的记忆修改范围 |
| 审核只拼接共同描述同一个且不可独立更新的事实。详细规则见 `design_detailed.md` 1.2.4 节 | 数据集或真实查询表明紧密逻辑、因果或时序关系因分散存储而难以召回和使用 | 实验更大的拼接粒度，允许将不同但紧密关联的事实组成更丰富的 memory unit，并配套更复杂的更新措施，与当前方案比较 |
| subject 达到 memory 数量或 content 字符软阈值后尝试 split；结果可以是 full split、partial split 或有明确理由的 defer split，实验版不设 subject 硬容量。详细规则见 `design_detailed.md` 1.2.5 节 | split 后仍频繁超限、结果碎片化、分组高度重叠、重要跨域 link 丢失，或参数无法跨 workload 泛化 | 调整触发阈值、最小记忆数量、结果 subject 规模目标、命名与多归属规则 |
| 公开 `search` 为每个 subject 同时维护 name 与 name + summary embedding，默认使用 name embedding。详细规则见 `design_detailed.md` 1.1.7 和 1.3.1 节 | Subject 召回质量不足，或 summary 能补充 name 无法表达的检索信号 | 对比两种 subject embedding，并按数据集问题类型分析收益与 summary 陈旧带来的影响 |
| `search` 的 Subject/Memory 通道分别召回 top-5/top-15，并为每个候选附带一个关联实体。详细规则见 `design_detailed.md` 1.3.1 节 | 固定 top-k 或 attached entity 数造成召回不足、冗余结果或上下文规模失衡 | 调整两个通道的 top-k 与 attached entity 数并按问题类型比较 |
| 第一版 search 使用 subject/memory 最低相似度 0.25/0.35，不设置最终输出字符数上限。详细规则见 `design_detailed.md` 1.3.1 节 | 阈值造成相关结果遗漏，或返回长度挤占主 Agent 上下文 | 评估相似度门槛、最终字符数上限和超限裁剪策略 |


实验版完整实现 memory extraction、Subject linking、Subject review、Subject split 和
search 后，再使用 `LongMemEval-S` 和 `LoCoMo_refined` 执行端到端评测。实验只按既定指标
输出结果并形成当前方案的基线；ablation study 和表中的扩展留到以后，在对应问题或收益能
通过数据集、真实 Agent 交互、benchmark 或 profiling 观察时再开展。实验执行与报告口径见
`design_detailed.md` 1.2.6 节。

### 工程工具


| 当前工具/方案                   | 需要扩展或替换的条件                                                                | 大致演进方向                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `pytest`                  | 出现明显不同的测试层级、外部系统依赖、随机状态空间或性能回归风险                                          | 保留 `pytest`，增加 markers、integration/end-to-end tests、property-based tests 和 benchmark/performance tests |
| `pytest-cov`，无硬覆盖率门槛      | 项目已有稳定测试分类和历史 coverage 基线，且未覆盖代码反复导致回归                                    | 对关键 core package 使用 branch coverage 和渐进式门槛；必要时接入 coverage report service，不用全局百分比代替风险判断                 |
| Ruff formatter + linter   | 团队规则增加或 Ruff 无法覆盖某项必要检查                                                   | 优先增加 Ruff rules；只有明确缺失时补充专用检查工具，不因项目变大自动替换 Ruff                                                        |
| `mypy`                    | 类型检查时间、第三方 typing 兼容性或诊断能力持续阻碍开发，且替代工具在本项目验证更好                            | 先调整模块范围和严格度；再通过试运行比较 Pyright 等替代方案，避免长期并行维护两套冲突规则                                                      |
| 单一 `dev` dependency group | 文档、benchmark、发布或 connector 开发依赖明显变大，并导致普通开发安装缓慢或依赖冲突                      | 拆分为 `test`、`lint`、`docs`、`benchmark` 等 dependency groups，并由 `dev` 组合常用组                                |
| 直接运行 `uv run ...`         | 命令组合频繁重复、参数难以记忆，或本地和 CI 编排开始漂移                                            | 先提供少量统一脚本或任务入口；只有需要多环境矩阵和复杂任务图时再考虑 `tox`/`nox` 等工具                                                     |
| 无 `pre-commit`            | 多人贡献后，格式和低成本检查经常直到 CI 才失败，显著增加反馈时间                                        | 增加只运行快速、确定性检查的 `pre-commit` hooks；耗时测试仍留在 CI                                                           |
| `uv_build` + pure Python  | FluxFold 自有代码经 profiling 证明需要 native extension，或 build backend 缺少已出现的必要能力 | native 路径评估 Maturin、scikit-build-core 或独立 acceleration package；其他 backend 只在有具体能力缺口时比较                 |


这些方向是演进路线而不是预先承诺。任何替换都需要先证明当前方案无法以较小扩展解决实际问题。
