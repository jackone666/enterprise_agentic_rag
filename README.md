# Developer Intelligent Customer Service Agent

面向开发者门户 / 企业技术支持 / 智能客服入口的多智能体 Agentic RAG 系统，可用于承接类似开发者官网右下角“智能客服”的问答、文档检索、代码示例生成、错误诊断、工单查询与人工兜底升级场景。

> 简历定位：开发者智能客服 Agent 平台。用户在官网悬浮客服入口输入问题后，系统通过 Deep Intent 识别请求类型，按需调用知识库检索、工具查询、代码生成/执行与答案校验能力，最终返回带引用、可追踪、可反馈的可信回答。

## ⚠️ 关键认知点（先看这个再读项目）

> 下面 10 条是面试/读代码最常被问错的认知陷阱，完整版见 `technical_deep_dive/` 各文档"易误会点"段。

### 认知 1：5 个 Agent ≠ 5 个 RetrievalMode

| 维度 | 数量 | 含义 |
|------|------|------|
| Agent | **5** | 1 Master + 4 Workers（master/tool/knowledge/code/verifier）|
| RetrievalMode | **5** | hybrid_only / parallel / graph_first / error_first / code_first |
| IntentCategory | **10** | 10 种用户意图分类 |

三者**正交**，不是 1-to-1-to-1。详见 `technical_deep_dive/主题/02-Agent设计.md` §四.5。

### 认知 2：5 个 RetrievalMode 全部走 RAG

模式只是"召回策略"（权重 + 是否走图谱），**不是"哪个 Agent 处理"**。`ToolAgent` 由 `MasterAgent._requires_tools()` 独立判断。

### 认知 3：AgentState 字段数 = 72

不是 ~67/~70 等约数，是 **72 个 TypedDict 字段**。详见 `technical_deep_dive/01-项目总览与系统架构.md` §1.1。

### 认知 4：max_graph_steps = 18（上限，不是实际步数）

典型请求 5-8 步完成，> 18 步才触发 `human_fallback`。

### 认知 5：5 级降级链（不是动态决策）

每级降级路径**预先定义**：Milvus 挂 → ES 关键词 → GraphRAG → human_fallback；Rerank 挂 → RRF 截断；LLM 挂 → fallback provider → mock。

### 认知 6：6 个 BaseTool 全部在 ToolAgent 单例里

| tier | 含义 | 例子 |
|------|------|------|
| safe | 直接执行 | get_system_status |
| sensitive | 需用户确认 | create_ticket / execute_code |
| destructive | 禁止 | (未实现) |

tier 是"用户介入程度"，**不是风险等级**。

### 认知 7：长期记忆分两类

| 类型 | 内容 |
|------|------|
| Semantic | 稳定偏好 / 技术栈 / 业务规则 |
| Episodic | 历史事件 / 任务结果 / 排障结论 |

### 认知 8：Harness 5 子系统

不是"加 GitHub Actions"，是 **5 个独立子系统**：离线评估 / 在线反馈 / 自动回滚 / 失败沉淀 / Eval Gate。

### 认知 9：Eval Gate 8 条 ≠ 全量评估

Eval Gate 是**门禁级**（关键 case），**全量评估**在 nightly/weekly 跑。

### 认知 10：项目目录结构反映了 v3.0 / v3.1 P0 修复

- `agents/deep_intent/`（不是 `agent/deep_intent/`）
- v3.0：`retrieval/` 下不再有 `retrieval_router.py` 和 `__init__.py`（已下线）
- **v3.1**：`retrieval/` 包**整体并入 `rag/`** —— `keyword_search_tool.py` / `vector_search_tool.py` / `graph_search_tool.py` / `merger.py` / `reranker_wrapper.py` / `evidence_selector.py` / `unified_schemas.py` 全部移到 `rag/`，`retrieval/` 目录已删除
- **v3.1**：`graph/workflow.py` 从 **1080 行**拆为 `graph/workflow.py`（~30 行 re-export 入口）+ `graph/builder.py`（图结构）+ `graph/nodes/`（16 个节点按职责拆分：`memory.py` / `permission.py` / `intent.py` / `master.py` / `retrieval.py` / `tools.py` / `context.py` / `generation.py` / `code.py` / `verify.py` / `finalize.py`）+ `graph/cache.py` + `graph/persistence.py` + `graph/dependencies.py`
- **v3.1**：`MasterAgent` / `MasterDecision` / `AgentState` 新增 `routing_path` 字段（`"llm"` / `"rule"` / `"rule_direct"`），可观测 LLM 路由器实际跑了多少次
- `last_worker` 字段值：`context_manager` / `knowledge_agent` / `code_agent`（不是 `answer_agent`）

---

## 🔑 5 个项目核心数字

| 数字 | 含义 |
|------|------|
| **5** | Agent 数 / RetrievalMode 数 |
| **10** | IntentCategory 数 |
| **16** | LangGraph Node 数 |
| **72** | AgentState 字段数 |
| **22** | Agent 决策评估用例数 |

---

## 📚 文档导航

| 文档 | 用途 |
|------|------|
| `technical_deep_dive/01-项目总览与系统架构.md` | 5 层架构、组件矩阵 |
| `technical_deep_dive/02-LangGraph-工作流编排.md` | 16 节点图结构（v3.1 拆分后位置） |
| `technical_deep_dive/03-Agent-体系与进阶设计.md` | 5 Agent + 决策矩阵 + 置信度公式 + `routing_path` 字段 |
| `technical_deep_dive/04-RAG-检索引擎与GraphRAG.md` | 检索链路 + 知识库更新（v3.1 删 `retrieval/` 包装层） |
| `technical_deep_dive/主题/` | 12 个主题题库（按面试主题复习）|
| `.claude/skills/doc-code-sync/SKILL.md` | 双向同步 skill |
| `.claude/CLAUDE.md` | 项目级 Claude Code 配置 |


## 原项目名

Enterprise Agentic RAG Multi-Agent QA System

面向企业内部知识库 / 开发者支持 / 客服问答场景，基于 LangGraph 实现的多智能体 Agentic RAG 问答系统。

## 业务场景包装

该系统可作为开发者官网智能客服底座，覆盖开发者常见咨询链路：

- 文档问答：解释 ArkUI 生命周期、API 参数、权限配置、发布流程等官方文档问题。
- 代码辅助：根据开发者问题检索 API 示例，生成 TS/JS/Python/Bash/ArkTS 代码，并进行沙箱执行验证。
- 错误诊断：针对错误码、构建失败、接口调用失败等问题，优先检索错误库、FAQ、历史工单和外部知识源。
- 迁移与兼容：通过 GraphFirst 检索 API 关系、版本差异和迁移路径，标注冲突证据。
- 工单与人工兜底：对低置信度、权限相关或高风险问题进入 ToolAgent / human_fallback 流程。
- 运营闭环：通过 Trace、反馈采集、Badcase 自动沉淀和 Eval Gate 持续优化客服回答质量。

## 核心能力

### Agent 体系
- **主从 Agent 架构** — MasterAgent(路由) + ToolAgent(工具) + Answer/Knowledge 生成能力 + CodeAgent 能力 + VerifierAgent(校验)
- **Deep Intent 深度意图识别** — 10 种意图分类 + 5 种检索模式 + 多级置信度
- **Claim-level 校验** 🆕 — 断言级幻觉检测，6 种断言类型逐条对照源文档
- **MasterAgent 路由** — LLM 优先 + 规则兜底，支持 10 个下游节点分发

### 检索体系
- **Adaptive RAG** — 向量 + 关键词 + 图谱 + 外部知识源四路融合
- **意图感知检索工作流** 🆕 — HybridRAG / GraphFirst / ErrorFirst / CodeGeneration 四种模式自动分派
- **Cross-Encoder 精排** 🆕 — Ollama qwen3-reranker-0.6b 替换规则重排序
- **外部知识源补充** — GitHub Issues / Stack Overflow / Web Search 多源检索
- **语义缓存** 🆕 — 双层缓存(精确匹配+Embedding相似度)，热门问题零延迟

### 质量保障
- **冲突证据检测** 🆕 — 多文档版本/否定/API废弃/建议冲突自动标注
- **Self-RAG 质量门控** — 文档评分 → 幻觉检测 → 答案校验
- **Eval Gate CI** 🆕 — 每次变更自动跑评估集，防止回归上线
- **Prompt Registry** 🆕 — 版本管理 + A/B 灰度 + 一键回滚
- **Agent 决策评估集** 🆕 — 22 条用例评估 MasterAgent 路由准确性

### 工程能力
- **流式输出 SSE** 🆕 — `/chat/stream` 端点实时推送工作流节点事件
- **深度思考（CoT Visible）** 🆕 — 答案生成前展示 AI 推理链，支持开关控制
- **智能客服前端 Widget** 🆕 — 独立全页模式 + 可嵌入浮动挂件，模仿华为开发者官网智能客服体验
- **OpenTelemetry 全链路追踪** 🆕 — OTLP 导出 + Jaeger/Tempo 可视化
- **Prometheus 告警** 🆕 — 8 组告警规则(延迟/错误率/检索质量/校验/系统)
- **自动回滚阈值校准** 🆕 — EWMA + 3σ 统计动态调整安全阈值
- 安全工具分级（Safe / Sensitive / 人工审批）
- 人工兜底与升级机制
- 多层级记忆系统（短期 / LLM摘要 / 用户档案 / 长期记忆跨会话持久化）
- 上下文管理与 Token 预算
- 故障恢复（重试 / 回退 / 降级）

详细文档：[开发者场景增强功能](docs/developer-features.md) | [TECHNICAL_DEEP_DIVE](TECHNICAL_DEEP_DIVE.md)

## 多 Agent 架构设计 🆕

### 架构模式：Master-Slave + Blackboard（主从 + 黑板）

系统采用 **Hub-and-Spoke 星型拓扑**。核心原则：**Agent 之间不直接通信**，所有交互通过共享 `AgentState`（72 字段黑板）和 `MasterAgent` 中央路由完成。

```
                    ┌──────────────────┐
                    │   MasterAgent     │
                    │  (中央路由调度器)   │
                    │ LLM优先 + 规则兜底  │
                    └───────┬──────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  ToolAgent  │   │ Knowledge   │   │  CodeAgent  │
│  工具调用    │   │  知识问答    │   │  代码生成    │
│  安全策略    │   │  检索+生成   │   │  +沙箱执行   │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
                ┌─────────────┐
                │VerifierAgent│
                │  答案校验    │
                │ Claim-level │
                └─────────────┘
```

### 为什么选主从架构

| 候选模式 | 判断 | 适用场景 |
|---------|------|---------|
| 去中心化（Peer-to-Peer） | ❌ | 自由协商型任务，本项目路径可预先穷举 |
| 分层规划（Plan-and-Execute） | ❌ | LLM 从零规划不可靠，额外验证开销大 |
| **主从（Master-Slave）** | ✅ | 中央路由确定性控制，Slave 解耦可测 |
| **黑板模式（Blackboard）** | ✅ | 72 字段 State 天然是共享上下文，零 RPC |

### MasterAgent 双模路由

```python
class MasterAgent:
    async def decide(self, state) -> MasterDecision:
        llm_result = await self._llm_decide(state)   # 1️⃣ LLM 优先
        if llm_result is not None:
            return llm_result
        return self._rule_decide(state)               # 2️⃣ 规则兜底
```

- **LLM 路由**：将意图、置信度、检索质量、工具错误、重试次数、代码状态等全部上下文信号打包成 prompt，LLM 输出 `{"next_node": "...", "reason": "..."}`
- **规则路由**：确定性 `_after_{last_step}()` 方法链，LLM 不可用时自动切换，保证每次请求都有确定路径

### Slave Agent 分工

| Agent | 核心能力 | 写入 State 的关键字段 |
|-------|---------|---------------------|
| **ToolAgent** | 工单查询、系统状态、用户信息、错误码查询 | `tool_results`, `tool_errors` |
| **Knowledge** | 向量+关键词+图谱融合检索、语义缓存、Cross-Encoder 精排、上下文构建、答案生成 | `retrieved_docs`, `draft_answer`, `citations` |
| **CodeAgent** | AST 符号提取、LLM 代码生成、六层安全沙箱执行 | `code_snippet`, `code_verified` |
| **VerifierAgent** | Claim-level 断言拆分、逐条对照源文档、幻觉检测 | `verified`, `verification_reason` |

**为什么拆分而不是合并：** 一个 Agent 管所有事 → 知识问答时误调工单、错误诊断时走错检索路径。职责分离让 Prompt 更短、工具集更聚焦、决策边界更清晰。

### 共享 State 黑板通信

Agent 之间**零直接 RPC 调用**，全部通过 `AgentState`（72 字段 TypedDict，15个功能区）读写：

```
AgentState 功能分区:
├── 路由层    master_next, master_reason, master_decisions, last_agent_step
├── 意图层    deep_intent, primary_intent, confidence, retrieval_mode
├── 检索层    retrieved_docs, reranked_docs, retrieval_backend
├── 工具层    tool_results, tool_errors, pending_confirmations
├── 代码层    code_snippet, code_verified, code_execution_result
├── 生成层    draft_answer, citations, structured_context
├── 校验层    verified, verification_reason, claim_results
├── 记忆层    chat_history, session_summary, user_profile, long_term_memories
├── 降级层    fallback_reason, recovery_action, retry_count, retry_history
└── 观测层    node_events, tool_events, retrieval_events, verification_events
```

**效果：** 新增一个 Agent → 只需 State 加字段 + MasterAgent 加路由分支 + Workflow 加节点，不触及已有 Agent 代码。

### 完整执行链路

```
POST /chat
  → load_memory → check_permission
      ├─ 拒绝 → final_refusal → save_memory → END
      └─ 通过 → deep_intent_recognition → master_agent
                  │
   ┌──────────────┴──────────────────────────────┐
   │  Hub-and-Spoke 调度循环:                    │
   │  call_tools / retrieve / rewrite / context  │
   │  generate_code / execute_code / answer      │
   │  verify_answer / finalize / human_fallback  │
   │  (每个 Slave 执行完必须回到 MasterAgent)     │
   │  安全阀: max_graph_steps=18                 │
   └─────────────────────────────────────────────┘
                  │
                  ▼
            save_memory → ChatResponse
```

## 结合项目的问答路径

不同问题会先进入 `deep_intent_recognition`，再由 `MasterAgent` 结合意图、置信度、工具需求和检索结果决定后续路径。

| 用户问题类型 | Deep Intent | 执行路径 | 检索策略 | 校验方式 |
|--------------|-------------|----------|----------|----------|
| "解释一下 ArkUI 生命周期" | `concept_qa` | 检索服务 → AnswerAgent → VerifierAgent | HybridRAG，vector-heavy | Claim-level |
| "给我一个调用 X API 的代码示例" | `code_generation` | 检索服务 → CodeAgent → 沙箱执行 → AnswerAgent | CodeGeneration 工作流，优先示例+API | 代码沙箱执行 |
| "这个错误码怎么排查？" | `error_diagnosis` | ToolAgent 查状态/工单 → 检索服务 → AnswerAgent | ErrorFirst 工作流，优先错误库+FAQ | Claim-level |
| "模块 A 到 B 怎么迁移？" | `migration` | 检索服务 → AnswerAgent → VerifierAgent | GraphFirst 工作流，优先图谱+迁移路径 | 冲突检测 |
| "查一下我的工单状态" | `project_debug` | ToolAgent → 检索服务 → AnswerAgent | 工具结果为主 + 知识库解释 | 规则兜底 |

知识库在当前架构中的定位是**证据来源**，不是独立 Agent。它会影响回答内容、引用和校验结果；真正决定"下一步去哪"的是 `MasterAgent`。

## 检索架构升级

### 意图感知工作流分派

```
retrieve_knowledge 节点
  │
  ├── 语义缓存命中 → 直接返回 (P95 ~10ms)
  │
  ├── hybrid_only / parallel → HybridRAGWorkflow
  │     └── 关键词(kw) + 向量(vec) 并行 → RRF 融合 → 重排 → 证据选择
  │
  ├── graph_first → GraphFirstWorkflow
  │     └── 图谱 → 查询扩展 → kw + vec 并行 → 三路融合 → 重排
  │
  ├── error_first → ErrorFirstWorkflow
  │     └── 错误库 → FAQ/工单 → kw → 融合 → 重排
  │
  ├── code_first → CodeGenerationWorkflow
  │     └── 示例 → API参考 → 官方文档 → 融合 → 重排
  │
  ├── 回退层1: GraphRAG Orchestrator
  ├── 回退层2: 旧版 Retriever (kw fallback)
  └── 外部增强: GitHub Issues + Stack Overflow + Web Search
```

### 重排序链

```
Cross-Encoder (qwen3-reranker-0.6b) → API Reranker → 规则(关键词+来源多样性)
       ↑ Ollama 本地推理                        ↑ 可配置端点        ↑ 始终可用
```

## 基础设施（Docker Compose）

系统依赖以下本地服务，通过 Docker Compose 一键启动：

| 服务 | 端口 | 用途 |
|------|------|------|
| PostgreSQL 16 | 5432 | 用户信息、会话记录、审计日志、评估数据、用户反馈 |
| Redis 7 | 6379 | 会话缓存、语义缓存、工具临时状态、rate limit |
| Milvus 2.4 | 19530 | 向量检索、HNSW 索引、metadata 过滤 |
| Elasticsearch 8 | 9200 | 全文关键词检索 (BM25)、IK 中文分词 |
| Neo4j 5.x | 7474/7687 | 知识图谱存储与检索 |
| MinIO | 9000/9001 | 原始文档存储、上传文件、离线评估文件 |
| OpenTelemetry Collector | 4317/4318 | Trace/Metric/Log 采集管线 |
| Prometheus | 9090 | 指标采集与存储 |
| Grafana | 3000 | 指标看板 (admin / admin_dev) |

### 启动基础设施

```bash
cp .env.example .env              # 首次需要，密码等配置在 .env 中
docker compose up -d               # 启动所有服务
./scripts/healthcheck.sh           # 验证所有服务就绪
```

### 管理脚本

```bash
./scripts/start_dev.sh             # 创建 .env + 启动 + 健康检查
./scripts/stop_dev.sh              # 停止所有服务（保留数据）
./scripts/reset_dev.sh             # 完全重置（删除数据 + 重新初始化）
./scripts/healthcheck.sh           # 检查 9 个服务是否正常
```

### 数据管理脚本

```bash
python scripts/init_db.py          # 初始化 PostgreSQL 数据库
python scripts/ingest_docs.py      # 文档入库（向量 + 关键词 + MinIO）
python scripts/ingest_graph.py     # 知识图谱构建
python scripts/build_graph_indexes.py  # 初始化 Neo4j 约束和索引
python scripts/run_eval_gate.py    # 运行 Eval Gate 评估门禁
```

### 无 Docker 环境

如果 Docker 服务未启动，系统**不会崩溃**，而是自动回退到内存 mock 实现：

| 服务 | 回退方案 |
|------|----------|
| PostgreSQL | → 内存 dict / Mock Repository |
| Redis | → 内存 dict / deque |
| Milvus | → MemoryVectorStore / Jaccard 关键词匹配 |
| Elasticsearch | → Jaccard 关键词匹配（Retriever 兜底） |
| MinIO | → 本地文件系统 |
| Neo4j | → 自动退回 hybrid_only 模式 |
| OpenTelemetry | → JSONL 文件 |
| Prometheus | → MetricsCollector 内存累计 |

## LLM Provider 配置

默认使用 mock LLM，无需 API key。升级至真实 LLM：

```bash
# Mock 模式（默认）
LLM_PROVIDER=mock

# OpenAI / vLLM / Ollama 等 OpenAI-compatible 接口
LLM_PROVIDER=openai-compatible
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1

# 阿里云 DashScope
LLM_PROVIDER=dashscope
LLM_MODEL=qwen-turbo
LLM_API_KEY=sk-xxxxxxxx
```

**Fallback 保证：** 真实 LLM 调用失败时自动回退到 mock，主流程不崩溃。

## Cross-Encoder 重排序 🆕

使用 Ollama 部署 `pdurugyan/qwen3-reranker-0.6b-q8_0` 模型进行 Cross-Encoder 精排：

```bash
# 安装 Ollama 并拉取模型
ollama pull pdurugyan/qwen3-reranker-0.6b-q8_0:latest

# 配置 .env
RERANKER_ENABLED=true
RERANKER_MODEL=pdurugyan/qwen3-reranker-0.6b-q8_0:latest
OLLAMA_BASE_URL=http://localhost:11434
RERANKER_TIMEOUT=30.0
RERANKER_BATCH_SIZE=20
```

**Fallback：** Ollama 不可用时自动回退到 API 重排 → 规则重排。

## 语义缓存 🆕

```bash
# .env 配置
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_TTL=3600       # 缓存 1 小时
SEMANTIC_CACHE_SIMILARITY=0.92 # 语义相似度阈值
SEMANTIC_CACHE_MAX_ENTRIES=1000
```

**双层级：** 精确匹配(SHA256) → 语义匹配(Embedding Cosine Similarity)

## Graph-Augmented Hybrid RAG

在原有 Hybrid RAG（关键词 BM25 + 向量检索）基础上，新增**知识图谱增强检索**能力。

### 路由策略

| 模式 | 触发条件 | 权重分配 | 说明 |
|------|----------|----------|------|
| `parallel` | 默认普通问题 | kw:0.3 vec:0.5 graph:0.2 | 三路并行召回 |
| `graph_first` | 关系类/迁移/兼容性问题 | kw:0.2 vec:0.3 graph:0.5 | 先图谱→扩展查询→再kw+vec |
| `error_first` | 错误诊断/调试问题 | error:0.4 faq:0.25 kw:0.2 ticket:0.15 | 优先错误知识库 |
| `code_first` | 代码生成/API用法 | sample:0.4 api:0.35 official:0.25 | 优先代码示例 |
| `hybrid_only` | 图谱不可用/关闭 | kw:0.4 vec:0.6 | 完全退化 |

### 降级保证

1. Graph RAG 关闭 → hybrid_only
2. Neo4j 不可用 → hybrid_only
3. 图谱为空 → 继续 kw + vec
4. 单路检索失败 → 其他路正常执行
5. Ollama 不可用 → 规则重排

## Prometheus + Grafana 集成

```bash
# Prometheus 指标端点
curl http://localhost:8000/prometheus_metrics

# JSON 业务指标
curl http://localhost:8000/metrics
```

**告警规则：** 8 组规则覆盖延迟/错误率/检索/校验/系统，规则文件 `deploy/prometheus/alerts.yml`

## 快速开始

```bash
# 安装后端依赖
make install

# 准备环境
cp .env.example .env

# 启动后端 (端口 8000)
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000
```

### 开发者控制台（原前端）

```bash
cd frontend
npm install
npm run dev
```

开发者控制台默认访问 http://localhost:5173 ，提供完整的 Agent Trace、RAG 证据面板、工具调用详情、Metrics 仪表盘、文件上传和告警面板。

### 智能客服前端（Widget）🆕

面向终端用户的智能客服界面，模仿华为开发者官网右下角智能客服交互体验。支持两种模式：

```bash
cd widget
npm install
npm run dev
```

| 模式 | URL | 说明 |
|------|-----|------|
| 全页模式 | http://localhost:5174 | 独立智能客服页面，含欢迎问候 + 推荐问题 + 聊天界面 |
| 浮动挂件 | http://localhost:5174/?mode=embedded | 右下角悬浮按钮 + 侧滑聊天面板，可嵌入任意网页 |

**Widget 功能对照：**

| 华为智能客服特性 | Widget 实现 |
|-----------------|------------|
| 欢迎页 + 推荐问题点击 | `WelcomeScreen` + `/api/suggestions` 端点 |
| 深度思考开关 | `DeepThinking` 折叠面板 + `deep_thinking` SSE 事件流 |
| ContentEditable 输入框 | `ChatInput`（Enter 发送，Shift+Enter 换行） |
| 复制 / 有帮助 / 没帮助 | `FeedbackButtons` + `/feedback` API |
| AI 生成免责声明 | 每条回复底部自动标注 |
| 结构化回答含代码示例 | 后端 CodeAgent + knowledge_agent |
| 右下角悬浮入口 | `FloatingWidget` 浮动按钮动画 |

### 流式输出 🆕

```bash
# SSE 流式端点 — 实时推送工作流节点事件 + 深度思考 + 回答流
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "HarmonyOS 生命周期是什么？", "user_id": "u001", "session_id": "demo", "deep_thinking": true}'
```

SSE 事件类型：

| 事件 | 说明 |
|------|------|
| `start` | 请求已接收，返回 trace_id |
| `node_end` | 工作流节点执行完成（节点名、延迟、成功/失败） |
| `thinking` | 深度思考推理链片段（仅 `deep_thinking: true` 时） |
| `answer_chunk` | 回答文本增量流式推送 |
| `done` | 处理完成，含完整答案、引用、校验结果 |
| `error` | 超时或异常 |
| `end` | 流终止 |

## Claude Code Skills

项目内置 8 个 Claude Code Skill，将常见开发操作封装为 `/命令` 一键执行：

| Skill | 命令 | 用途 |
|-------|------|------|
| **setup-dev** | `/setup-dev` | 开发环境一键启动：Docker → 健康检查 → DB 初始化 → 文档入库 → 图谱构建 → 后端/前端启动 |
| **health** | `/health` | 完整 9 服务健康检查（PG、Redis、Milvus、MinIO、ES、Neo4j、Prometheus、Grafana、OTel） |
| **reset-env** | `/reset-env` | 一键销毁所有数据卷 + 重建环境 + 数据恢复 |
| **ingest-all** | `/ingest-all` | 全量数据入库管道（文档 → ES + Milvus + MinIO + Neo4j 图谱） |
| **run-tests** | `/run-tests` | 按类型运行测试（单元 `-m "not integration"` / 集成 / 全量），自动检查依赖服务 |
| **eval-gate** | `/eval-gate` | 运行评估门禁（8 个内置 case），输出 faithfulness/recall/relevancy/intent 四项指标 |
| **analyze-failures** | `/analyze-failures` | 导出失败案例 → 归类失败模式（意图/检索/幻觉）→ 给出改进方向 |
| **debug-retrieval** | `/debug-retrieval "查询"` | 对指定查询输出完整检索 trace（命中数/延迟/融合权重/降级链路） |

### 使用方式

在 Claude Code 对话中直接输入 `/命令名`：

```bash
/health                          # 检查所有服务状态
/run-tests                       # 运行全量测试
/debug-retrieval "Ability 和页面生命周期有什么关系？"  # 调试检索
```

Skill 文件位于 `.claude/skills/` 目录，每个 skill 包含完整的实现指导和常见问题排查表。

## 兜底与恢复机制

系统在任何节点失败时都能给出稳定兜底，绝不崩溃或胡说。

### 兜底类型

| 类型 | 触发条件 | 默认动作 | 可恢复 |
|------|----------|----------|--------|
| `permission_denied` | 用户缺少 `knowledge_search` 权限 | `final_refusal` | ❌ |
| `no_relevant_docs` | 检索返回空结果 | `rewrite_query` | ✅ |
| `low_retrieval_score` | 所有文档评分 < 0.1 | `use_keyword_retriever` | ✅ |
| `tool_failure` | 工具执行返回错误 | `retry` | ✅ |
| `answer_not_grounded` | 答案校验未通过（含幻觉断言） | `regenerate_answer` | ✅ |
| `llm_failure` | LLM 调用失败 | `retry` | ✅ |
| `unknown_intent` | 意图无法识别 | `human_fallback` | ✅ |

### 重试策略

| 节点 | 最大重试 | 说明 |
|------|----------|------|
| `retrieve_knowledge` | 1 | 首次失败 → query rewrite → 再检索 |
| `call_tools` | 2 | 工具失败后自动重试 |
| `generate_answer` | 1 | 校验失败后重新生成 |
| `verify_answer` | 1 | 校验失败 → regenerate → 再校验 |

### 自动回滚阈值 🆕

基于 EWMA + 3σ 统计动态校准回滚安全阈值，支持随时观测指标并自动判断是否需要回滚。

## 稳定性与性能 🆕

> 详细技术文档：[TECHNICAL_DEEP_DIVE §43](TECHNICAL_DEEP_DIVE.md#43-专题稳定性与性能)

系统从**限流、熔断、降级、缓存、并行化、超时控制**六个维度构建稳定性保障体系。核心原则：**故障隔离不扩散、Fail-open 优先可用、多层降级逐级退化、可观测驱动治理**。

### 六层降级链路

每一层失败自动滑入下一层，确保系统始终有可用路径返回结果：

```
L0 语义缓存 (P95 ~10ms) → 命中直接返回
  ↓ 未命中
L1 意图感知检索工作流 (4路并行 + Cross-Encoder 精排)
  ↓ 召回为空/低分
L2 GraphRAG Orchestrator (知识图谱扩展 + 三路融合)
  ↓ Neo4j 不可用/图谱为空
L3 旧版 Retriever 兜底 (关键词 + 简单向量)
  ↓ ES/Milvus 均不可用
L4 外部知识源增强 (GitHub Issues + Stack Overflow + Web Search)
  ↓ 外部 API 全部不可用
L5 规则兜底回答 (FAQ 模板 + 静态规则，终极兜底)
```

### 降级矩阵

| 目标服务 | 降级方式 | 影响范围 |
|----------|----------|----------|
| Cross-Encoder (Ollama) | → API Reranker → 规则排序 | 仅重排精度 |
| LLM (主 Provider) | → 备选 Provider → MockProvider | 回答质量 |
| PostgreSQL | → 内存 dict / Mock Repository | 会话持久化 |
| Redis | → 内存 dict / deque | 缓存/限流/会话 |
| Milvus | → MemoryVectorStore | 仅向量召回 |
| Elasticsearch | → Jaccard 关键词匹配 | 仅关键词召回 |
| MinIO | → 本地文件系统 | 文件存储位置 |
| Neo4j | → hybrid_only 模式 | 仅图谱增强 |
| OpenTelemetry | → JSONL 文件 | 链路追踪 |
| Prometheus | → MetricsCollector 内存 | 指标采集 |

### 熔断机制

LLM Provider 层实现**三级自动切换**，无需人工干预：

| 层级 | Provider | 触发条件 | 延迟 |
|------|----------|----------|------|
| L1 主力 | OpenAI / DashScope | 正常 | ~1-3s |
| L2 备选 | DashScope / vLLM / Ollama | L1 连续失败 | ~2-5s |
| L3 兜底 | MockProvider（模板化） | L2 连续失败 | ~10ms |

### 限流策略

基于 **Redis Lua 原子滑动窗口**实现租户级限流：

```bash
# .env 配置
RATE_LIMITER_MAX_PER_MINUTE=100  # 每租户每分钟最大请求数
RATE_LIMITER_FAIL_OPEN=false     # prod=false 被限流直接拒绝；dev=true 内存兜底
```

Redis 不可用时：**prod 环境拒绝请求**（保守防过载）；**dev 环境内存 dict 兜底**。

### 性能优化

**缓存层级：**

| 缓存层 | 命中率 | P95 延迟 | 原理 |
|--------|--------|----------|------|
| 精确匹配 (SHA256) | ~10% | < 1ms | 相同 query 直接返回 |
| 语义匹配 (Embedding 相似度 ≥0.92) | ~20% | ~10ms | 语义相近 query 命中 |
| LLM 响应缓存 | ~5% | < 1ms | 相同 prompt 复用 |

**并行化策略：**

| 操作 | 方式 | 加速比 |
|------|------|--------|
| 混合检索（kw + vec + graph） | `asyncio.gather` 三路并行 | ~2-3x |
| 多工具调用 | ToolAgent 并行编排 | ~2-4x |
| 外部搜索（GitHub + SO + Web） | `asyncio.gather` 三路并行 | ~2-3x |
| 批量 Rerank | Cross-Encoder batch_size=20 | ~5-10x |

**连接池：** Redis max_connections=5 + keepalive | PostgreSQL asyncpg pool_size=20 | ES 持久连接

**超时控制：** 整体请求 60s | LLM 调用 60s | Reranker 30s | 沙箱执行 15s | LangGraph 最大 18 步 | LLM 最大 6 次/请求

### 环境感知安全策略

`APP_ENV` 自动区分 development 和 production 行为，防止 dev 宽松策略泄漏到生产：

| 策略 | development | production |
|------|-------------|------------|
| 内存回退兜底 | ✅ 允许 | ❌ 禁止（显式报错） |
| 限流 fail-open | ✅ 内存兜底 | ❌ 直接拒绝 |
| 本地代码执行 | ✅ 允许 | ❌ 仅沙箱 |

详细配置参见 `config/settings.py` 的 `RuntimeSettings`。

## 可观测性

### 事件体系

| 事件类型 | 触发时机 | 写入目标 |
|----------|----------|----------|
| `node_start` / `node_end` | 每个 LangGraph 节点 | JSONL；OTel 节点 span 待接入 |
| `tool_call` | 每次工具执行 | JSONL + Metrics |
| `retrieval` | 每次知识库检索 | JSONL + Metrics |
| `verification` | 每次答案校验 | JSONL + Metrics |
| Claim-level 结果 | `verification` 内部执行 | 当前记录在 verifier 日志/校验原因中，尚未独立成 JSONL 事件 |

### OpenTelemetry 集成 🆕

```bash
# .env
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=enterprise-agentic-rag
```

`observability/otel_integration.py` 已提供 FastAPI/httpx 自动埋点和 `traced_span` 工具；当前主链路仍以 JSONL `traced_node` 为主，若要生产启用 OTel，需要在应用入口显式初始化并接入节点 span。

## 评估体系与 Data Flywheel

### Eval Gate CI 🆕

每次 PR 自动运行评估门禁：

```bash
python scripts/run_eval_gate.py --threshold 0.7
```

输出：`overall` / `faithfulness` / `context_recall` / `answer_relevancy` / `intent_accuracy`

### Agent 决策评估集 🆕

22 条测试用例覆盖全部 10 种意图，评估 MasterAgent 路由准确性：
- `intent_accuracy`：意图识别正确率
- `routing_accuracy`：路由分发正确率
- `mode_accuracy`：检索模式选择正确率

### Data Flywheel 循环

```
线上运行 → 失败沉淀 → 失败分析 → 改进实施 → 回归验证 → 上线
   ↑                                                       ↓
   └──────────────────── 迭代循环 ──────────────────────────┘
```

## 项目结构

```
enterprise_agentic_rag/
├── src/enterprise_agentic_rag/
│   ├── app/               # FastAPI 应用（/chat, /chat/stream, /feedback）
│   ├── agent/             # Deep Intent 深度意图识别
│   │   └── deep_intent/   # 10意图分类 + 5检索模式 + 置信度
│   ├── agents/            # 主从 Agent 与生成/校验能力模块
│   │   ├── master_agent.py      # 中央路由调度
│   │   ├── knowledge_agent.py   # 知识问答生成
│   │   ├── code_agent.py        # 代码生成 + AST
│   │   ├── tool_agent.py        # 工具编排执行
│   │   ├── verifier_agent.py    # 答案校验（含Claim-level）
│   │   └── claim_verifier.py    # 🆕 断言级幻觉检测
│   ├── rag/               # 检索引擎
│   │   ├── cross_encoder_reranker.py  # 🆕 Ollama qwen3-reranker
│   │   ├── semantic_cache.py          # 🆕 语义缓存
│   │   ├── graph_rag_orchestrator.py  # GraphRAG 编排
│   │   ├── fusion.py                  # 多路 RRF 融合
│   │   ├── retrieval_router.py        # 动态路由（v3.0 已下线，被 workflow 替换）
│   │   ├── graph/                     # 知识图谱（实体/关系/索引/检索）
│   │   ├── external/                  # 外部知识源检索
│   │   ├── observability/             # 检索链路追踪
│   │   ├── keyword_search_tool.py     # 🆕 v3.1 合并 retrieval/ 进来
│   │   ├── vector_search_tool.py      # 🆕 v3.1 合并
│   │   ├── graph_search_tool.py       # 🆕 v3.1 合并
│   │   ├── merger.py                  # RRF 融合（v3.1 合并）
│   │   ├── reranker_wrapper.py        # Cross-Encoder 优先（v3.1 合并）
│   │   ├── evidence_selector.py       # 证据选择（v3.1 合并）
│   │   └── unified_schemas.py         # 🆕 v3.1 工具 I/O 契约
│   ├── workflows/         # 意图感知检索工作流（v3.1 仍在，调用 rag/ 内的工具）
│   │   ├── hybrid_rag_workflow.py
│   │   ├── graph_first_workflow.py
│   │   ├── error_first_workflow.py
│   │   └── code_generation_workflow.py
│   ├── graph/             # LangGraph 工作流（v3.1 拆分）
│   │   ├── state.py                 # AgentState（72 字段 + v3.1 新增 routing_path）
│   │   ├── workflow.py              # ~30 行 re-export 入口
│   │   ├── builder.py               # 🆕 v3.1 图结构（节点 + 边）
│   │   ├── dependencies.py          # 🆕 v3.1 共享单例
│   │   ├── cache.py                 # 🆕 v3.1 缓存键 namespace
│   │   ├── persistence.py           # 🆕 v3.1 QA log 持久化
│   │   └── nodes/                   # 🆕 v3.1 16 节点按职责拆分
│   │       ├── memory.py            # load_memory / save_memory
│   │       ├── permission.py        # check_permission / final_refusal
│   │       ├── intent.py            # deep_intent_recognition
│   │       ├── master.py            # master_agent（带 routing_path）
│   │       ├── retrieval.py         # retrieve_knowledge / rewrite_query
│   │       ├── tools.py             # call_tools
│   │       ├── context.py           # build_context
│   │       ├── generation.py        # generate_answer + CoT
│   │       ├── code.py              # generate_code / execute_code
│   │       ├── verify.py            # verify_answer
│   │       └── finalize.py          # finalize_answer / human_fallback
│   ├── memory/            # 多层级记忆
│   │   ├── memory_manager.py        # 统一入口
│   │   ├── short_term_memory.py     # 短期记忆
│   │   ├── summary_memory.py        # 摘要记忆
│   │   ├── user_memory.py           # 用户记忆
│   │   ├── checkpoint.py            # 检查点
│   │   └── long_term_memory.py      # 长期记忆（已升级语义打分）
│   ├── context/           # 上下文管理
│   │   ├── context_manager.py       # 上下文构建
│   │   ├── token_budget.py          # Token 预算
│   │   ├── citation_manager.py      # 引用管理
│   │   ├── prompt_builder.py        # Prompt 模板
│   │   ├── prompt_registry.py       # 🆕 Prompt 版本管理
│   │   └── conflict_detector.py     # 🆕 冲突证据检测
│   ├── llm/               # LLM Provider 抽象层
│   ├── tools/             # 工具系统 + 安全策略
│   ├── storage/           # 数据持久化层
│   ├── middleware/        # 中间件（租户隔离/限流）
│   ├── recovery/          # 故障恢复
│   │   ├── recovery_manager.py
│   │   ├── fallback_policy.py
│   │   ├── retry_policy.py
│   │   └── threshold_calibrator.py  # 🆕 自动回滚阈值
│   ├── observability/     # 可观测性
│   │   ├── tracer.py
│   │   ├── event_schema.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── otel_integration.py      # 🆕 OpenTelemetry
│   │   └── prometheus_alerts.py     # 🆕 告警规则
│   ├── evals/             # 评估体系
│   │   ├── regression_eval.py
│   │   ├── rag_eval.py
│   │   ├── answer_eval.py
│   │   ├── agent_decision_eval.py   # 🆕 Agent 决策评估集
│   │   └── ...
│   └── config/
│       └── settings.py    # 16 个配置组（含 Reranker/Cache Settings）
├── frontend/              # 开发者控制台：React + TypeScript + Vite + Tailwind
├── widget/                # 🆕 智能客服前端：React + Vite + Tailwind（全页 + 浮动挂件）
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWidget.tsx        # 主聊天控件（page/embedded 双模式）
│   │   │   ├── WelcomeScreen.tsx     # 欢迎页 + 推荐问题网格
│   │   │   ├── ChatMessage.tsx       # 消息气泡 + CoT 链 + 引用
│   │   │   ├── ChatInput.tsx         # ContentEditable + 深度思考开关
│   │   │   ├── DeepThinking.tsx      # 折叠式推理面板
│   │   │   ├── FeedbackButtons.tsx   # 复制/点赞/点踩
│   │   │   └── FloatingWidget.tsx    # 右下角悬浮入口
│   │   ├── api/chat.ts              # SSE 流式 + 推荐问题 + 反馈 API
│   │   └── types/chat.ts            # 前端类型定义
│   └── dist/                        # 构建产物（可部署到 CDN）
├── tests/                 # 31 个测试文件
├── scripts/               # 运维脚本
│   └── run_eval_gate.py   # 🆕 Eval Gate 入口
├── .github/workflows/
│   └── eval-gate.yml      # 🆕 Eval Gate CI
├── deploy/prometheus/
│   └── alerts.yml         # 🆕 Prometheus 告警规则
├── docker-compose.yml     # 9 服务定义
└── pyproject.toml
```
