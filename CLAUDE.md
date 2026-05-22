# CLAUDE.md — Σ (Sigma) Multi-Agent Collaboration Framework

> 通用多智能体 AERC 协同框架。零领域知识内置，全部通过 Config + ToolSpec 注入。  
> 哲学：**底线质量 · 极致效率 · 可控成本**  
> 测试：**985 tests · 80% coverage**（2026-05-21）

## 项目定位

Σ 是一个可申请专利的通用多智能体协同框架。RocketFactory 是旗舰应用。

- **开源版 (MIT)**: 核心 AERC 引擎、交叉审查、收敛判断、成本追踪
- **企业版 (闭源)**: SSO/RBAC、HIPAA/SOC2、监控仪表板、SLA

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python ≥ 3.12 |
| LLM | OpenAI 兼容 API（DeepSeek V4 Pro 为主） |
| 存储 | SQLite（Memory 系统）+ JSON（checkpoint） |
| 测试 | pytest，目标 ≥ 80% 覆盖 |

## 目录结构

```
sigma/
├── CLAUDE.md                    # 本文件
├── README.md                    # 用户文档
├── pyproject.toml               # pip install -e .
├── src/sigma/
│   ├── __init__.py              # 公共 API 全部在此导出
│   ├── protocol.py              # AERC 协议引擎（核心，~1000行）
│   ├── orchestrator.py          # 循环调度 + resume/checkpoint + memory save
│   ├── state.py                 # 三层共享状态 + ConsensusEstimate + checkpoint
│   ├── convergence.py           # 收敛判断 + 振荡检测 + 物理不可能检查
│   ├── triggers.py              # 事件驱动触发器（知识缺口/工具异常/数据偏差）
│   ├── config.py                # SigmaConfig（全部领域知识由此注入）
│   ├── llm.py                   # LLM 后端（OpenAI 兼容 + 流式 + 重试）
│   ├── cache.py                 # LLM 缓存（sha256 + LRU + TTL）
│   ├── agent.py                 # Agent + BaseTool 抽象
│   ├── discovery.py             # 智能体/工具/技能 自动发现
│   ├── schema_validator.py      # JSON schema 递归校验 + 自动重试
│   ├── cost_tracker.py          # Token 成本追踪
│   ├── memory.py                # 跨会话 Memory 存储（SQLite）
│   ├── provenance.py            # 参数溯源 AuditTrail
│   ├── replay.py                # 回放测试模式
│   ├── hooks.py                 # 钩子系统
│   ├── log.py                   # 日志
│   ├── learning.py              # 执行记录 + 经验学习（CJK bigram tokenizer）
│   ├── observability.py         # OpenTelemetry 追踪（零依赖 fallback）
│   ├── knowledge.py             # 知识库引擎（TF-IDF + 可选 embedding）
│   ├── tool_cache.py            # 工具调用缓存（LRU + TTL）
│   ├── guardrails.py            # 参数守护（范围/跨参数/集合检查）
│   ├── pydantic_validator.py    # Pydantic 输出校验 + 自动重试
│   ├── tau/                # 🆕 Tau-led 层次化框架
│   │   ├── types.py             # 数据类型
│   │   ├── decomposer.py        # LLM 任务拆解
│   │   ├── executor.py          # 依赖感知并行执行
│   │   ├── detector.py          # 接口冲突检测
│   │   ├── resolver.py          # 三级分级调解（核心创新）
│   │   └── orchestrator.py      # 顶层编排
│   └── benchmark/               # 基准评测框架
│       ├── tasks.py             # 10 个标准化任务
│       ├── metrics.py           # 5 维度评分
│       ├── runner.py            # replay + live 执行器
│       └── reporter.py          # Markdown + JSON 报告
└── tests/
    ├── test_protocol.py         # 协议引擎测试（~250 tests）
    ├── test_protocol_async.py   # 异步协议测试
    ├── test_convergence.py      # 收敛判断测试
    ├── test_triggers.py         # 触发器测试
    ├── test_config.py           # 配置测试
    ├── test_llm.py              # LLM 后端测试
    ├── test_cache.py            # 缓存测试
    ├── test_schema.py           # Schema 校验测试
    ├── test_state.py            # 状态管理测试
    ├── test_cost_tracker.py     # 成本追踪测试
    ├── test_memory.py           # Memory 存储测试
    ├── test_provenance.py       # 溯源测试
    ├── test_benchmark.py        # 基准评测测试
    ├── test_tau.py               # 🆕 Tau 框架测试（130 tests）
    ├── test_learning.py          # 经验学习测试（24 tests）
    ├── test_observability.py     # 可观测性测试（15 tests）
    ├── test_knowledge.py         # 知识库测试（36 tests）
    ├── test_tool_cache.py        # 工具缓存测试（20 tests）
    └── test_discovery.py         # 自动发现测试
```

## 核心架构

### AERC 循环（SigmaProtocol）

```
A (Analyze) — 多角色并行独立分析 + 交叉审查（盲审）
E (Execute) — 并行工具调用（OpenAI function calling 优先，文本标记 fallback）
R (Review)  — 数据驱动的跨角色审查 + 魔鬼代言人
C (Converge)— 收敛判断 + 共识估算 + 知识沉淀（Skill Crafter）
```

### 复杂度自适应（三层分级）

| 层级 | 评分 | 角色数 | 轮次 | LLM 调用 | 成本 |
|------|------|--------|------|----------|------|
| LITE | ≤ 2.5 | 2-4 | 1 | ~8 | ~¥0.05 |
| STANDARD | ≤ 6.0 | 4-6 | 2-3 | ~30 | ~¥0.20 |
| RIGOROUS | > 6.0 | 8 | ≤ 4 | ~80 | ~¥0.50 |

纯规则引擎评估（零 LLM 调用），5 个维度：任务长度、领域覆盖、数值参数、约束关键词、行动动词。

### Tau 层次化框架（🆕）

```
TauOrchestrator: Decompose → Execute → Detect → Resolve → Iterate

Resolve 三级分级调解：
  Level 1 (DIRECT)  — 直接讨论，共享分析，协调员调解（小分歧）
  Level 2 (SIGMA)   — 聚焦盲审，独立估算 + 交叉审查（显著分歧）
  Level 3 (DIRECTOR) — 总监看全部证据，做方向性决策（僵持/严重分歧）

升级链: DIRECT → (失败) SIGMA → (失败) DIRECTOR
迭代 ≥ 3 轮: 跳过 DIRECT，直入 SIGMA/DIRECTOR
```

### 核心机制

- **盲审交叉审查**：角色独立分析（互不可见），然后互相审查——防止群体思维
- **共识估算**：工具不可靠时，多角色独立估算 → 互相审视 → 收敛为共识范围（带置信度 HIGH/MEDIUM/LOW）
- **收敛判断**：数值变化 < 5% + 无新定性冲突 → 收敛；振荡检测（2 轮无进展 → 硬停车）
- **魔鬼代言人**：安全官每轮从最坏情况审视（RIGOROUS 模式）
- **Skill Crafter**：检测知识缺口 → 搜索 → 生成技能文件，下轮注入（RIGOROUS 模式）
- **LLM 缓存**：sha256 内容寻址缓存 + LRU + TTL
- **流式输出**：实时 token 流 + 进度里程碑
- **Schema 校验**：递归校验 JSON 输出 + 自动重试
- **参数溯源**：每个值追踪来源（tool/agent_estimate/consensus/manual）
- **Checkpoint/Resume**：完整 SharedState 序列化，中断可恢复
- **经验学习**：执行记录持久化 + CJK bigram 相似检索 + lesson 注入 decomposer
- **可观测性**：OpenTelemetry 懒加载 + `@traced` 装饰器 + `tracing_span()` 上下文管理器
- **知识库**：TF-IDF 检索引擎 + 多格式摄入（txt/md/csv/json/pdf）+ 可选 embedding
- **工具缓存**：LRU + TTL 工具调用缓存，同步/异步 `get_or_run`
- **参数守护**：范围检查 + 跨参数检查 + 集合检查 + 自定义检查
- **OpenAI Function Calling**：`ToolSpec.to_openai_schema()` 从 `_run()` 签名自动生成 function 定义；`do()` 优先 function calling（LLM 动态参数）→ fallback 文本标记
- **大规模 LLM 失败检测**：P 阶段后自动检测，≥50% 智能体返回 `[LLM_ERROR]` → 立即中止（verdict="error"），防止空数据假收敛
- **推理模型流式支持**：`chat_stream()` 同步+异步版本均收集 `delta.reasoning_content`（DeepSeek V4 Pro 兼容）

## 关键命令

```bash
# 测试
pytest tests/ -v                           # 全部（当前 985 tests）
pytest tests/test_tau.py -v           # Tau 框架
python -m pytest tests/ -q                 # 快速回归
pytest --cov=sigma --cov-report=term       # 覆盖率（当前 80%）

# 类型检查
mypy src/sigma/

# 安装
pip install -e .
```

## 代码规范

- Python: PEP 8，类型注解，函数 < 50 行，文件 < 800 行
- 不可变模式优先（dataclass，返回新副本而非原地修改）
- 所有错误显式处理，禁止静默吞异常
- 测试覆盖率目标 ≥ 80%
- 提交格式: `feat: / fix: / refactor: / test: / chore:`

## 关键设计决策

1. **零领域知识内置**：火箭推进剂、医疗术语、金融模型——全部通过 `SigmaConfig` 注入
2. **工具鸭子类型**：任何有 `_run()` 的类就是工具，无需特定基类
3. **框架不可知 LLM**：任何 OpenAI 兼容 API 均可使用
4. **不可变状态**：每次状态更新返回新副本，安全并发
5. **分级调解 > 全量盲审**：Tau 框架中，小分歧直接沟通（DIRECT），大分歧才盲审（SIGMA），僵持上升总监决策（DIRECTOR）
6. **快路径优先**：流式 LLM + 并行共识估算 + 复杂度自适应 = 底线质量不妥协的前提下极致效率
7. **Function Calling > 文本标记**：工具调用优先原生 function calling（动态参数），fallback `[需要工具: xxx]` 文本匹配
8. **防御性故障检测**：大规模 LLM 失败自动检测 + 推理模型流式兼容 + 参数不匹配自动 fallback

## Sigma vs AutoGen 差距补齐计划

| 优先级 | 补齐项 | 改动量 | 状态 |
|--------|--------|--------|:----:|
| **P0** | Human-in-the-loop — AERC 阶段 HumanGate | ~150行 | ✅ |
| **P1** | Code Sandbox — AST 白名单 + subprocess 隔离 | ~200行 | ✅ |
| **P2** | Tau DIRECT 自由讨论 | ~100行 | ✅ |
| **P3** | Multimodal 协议支持 | ~50行 | ✅ |
| **P4** | GUI — 有意不做（CLI 优先是优势） | N/A | ❌ |

**设计原则**：不完全复制 AutoGen 的设计选择。AERC 结构化（盲审 + 收敛判断 + 共识估算）对工程任务优于 AutoGen 的自由 GroupChat。

## 已知待办

- [x] Tau 框架接入真实 LLM 端到端验证 (2026-05-20; 3 subtasks, 0 conflicts, ¥0.01, 1 round)
- [x] Tau 与 Sigma 统一入口（自动选模式） — `SigmaOrchestrator.run(mode="auto")` 自动路由
- [x] Tau 框架 skill 注入 — decomposer + resolver 可注入领域技能
- [x] Tau 框架 cost tracking — `CostTracker` 集成到 `TauOrchestrator`
- [x] Tau 框架 checkpoint/resume — `TauState.to_dict()` + `TauOrchestrator.checkpoint()` / `resume()`
- [x] Tau 框架 8 项技术完善 — CapabilityRegistry / Progressive Disclosure / Async Executor / Benchmark / Cache / Retry
- [x] PDCA → AERC 重命名 — 28 处修改，11 个文件
- [x] OpenAI → Universal 命名 — `UniversalBackend` + 向后兼容别名
- [x] 经验学习系统 — `LearningStore` + CJK bigram tokenizer + lesson 注入
- [x] 可观测性 — OpenTelemetry 懒加载 + `@traced` + `tracing_span()`
- [x] 测试覆盖达到 80% — 826 tests（知识库/工具缓存/自动发现/学习/可观测性）
- [x] **P0: Human-in-the-loop** — AERC 阶段 HumanGate（APPROVE/REVISE/OVERRIDE/REJECT）
- [x] **P1: Code Sandbox** — AST 白名单 + subprocess 隔离代码执行
- [x] **P2: Tau DIRECT 自由讨论** — 小冲突直接对话，不盲审
- [x] **P3: Multimodal 协议支持** — message content 支持 image
- [x] **真实场景验证** — 软件工程（短链服务）+ 医疗（降压药联合用药）双场景 AERC 端到端通过；非花架子
- [x] **OpenAI Function Calling 集成** — `do()` 优先 function calling + fallback 文本标记
- [x] **3 个鲁棒性 Bug 修复** — 大规模 LLM 失败检测 / `agent_analyses` 返回 / `reasoning_content` 收集
- [x] **专利决策确定** — 不申请；走 arXiv 防御性公开 + AGPLv3 + AERC 协议标准路线
- [ ] 国际化（中文 → 英文）
- [ ] arXiv 防御性公开投稿
- [ ] GitHub 仓库公开 + PyPI 发布
- [ ] #28: 完整火箭设计端到端验证
