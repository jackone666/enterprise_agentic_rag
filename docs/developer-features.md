# 开发者场景增强功能

> 面向开发者支持场景的增强能力：Code Agent、Symbol 级代码理解、示例代码优先检索、外部知识源补充检索、Cross-Encoder 精排、语义缓存、Claim-level 校验、冲突检测、Prompt Registry、流式输出。

---

## 1. Code Agent（代码生成与执行验证）

### 概述

当开发者询问"怎么调用 X API"时，系统会先由 `deep_intent_recognition` 识别为 `code_generation` 或 `api_usage`，再由 `MasterAgent` 调度检索服务和代码生成能力。Code Agent 能力负责**生成可运行的代码片段**并在**沙箱中执行验证**，确保返回的代码经过实际验证。

### 工作流

```
用户查询 → deep_intent_recognition (检测到 code_generation/api_usage)
         → master_agent
         → retrieve_knowledge (CodeGeneration 工作流：示例→API→官方文档)
         → master_agent
         → build_context
         → master_agent
         → generate_code (Code Agent 生成代码)
         → execute_code (沙箱执行)
              ├─ 执行成功 → generate_answer (注入已验证代码)
              ├─ 执行失败 + 可重试(max 2) → generate_code (修复重试)
              └─ 执行失败 + 耗尽 → generate_answer (带免责声明)
```

### 和项目问答的关系

| 问法 | 识别意图 | 检索工作流 | 后续动作 |
|------|----------|-----------|----------|
| "X API 怎么调用？" | `api_usage` | CodeGenerationWorkflow | 示例代码+API文档优先 |
| "给我 X 的示例代码" | `code_generation` | CodeGenerationWorkflow | 代码优先，生成并沙箱验证 |
| "这个报错怎么解决？" | `error_diagnosis` | ErrorFirstWorkflow | 先走 ToolAgent 查状态，再补充知识库 |
| "A 模块迁移到 B 怎么做？" | `migration` | GraphFirstWorkflow | 优先走图关系和迁移路径 |

### 支持的编程语言

- TypeScript / JavaScript / ArkTS（HarmonyOS）
- Python
- Bash / Shell

### 沙箱安全机制（六层纵深防御）

代码执行是系统中风险最高的操作。沙箱机制通过 **六层纵深防御** 确保代码生成和执行的安全性，涉及 6 个核心模块：

| 层 | 模块 | 职责 |
|----|------|------|
| L1 | `tools/policies.py` | **三级安全策略门控** — safe/sensitive/destructive 分级 |
| L2 | `tools/code_execution_tool.py` | **静态恶意模式黑名单** — 19 种危险模式匹配 |
| L3 | `config/settings.py` | **编程语言白名单** — 仅允许 5 种语言 |
| L4 | `config/settings.py` | **环境感知安全** — 生产环境默认禁止本地执行 |
| L5 | `tools/code_execution_tool.py` | **Subprocess 进程级隔离** — 临时文件 + 超时 + kill |
| L6 | `tools/executor.py` | **熔断器 + 重试 + 审计** — 5 连失败自动熔断 |

#### L1 — 三级安全策略门控

所有工具在 `ToolExecutor.execute()` 调用前必须先过 `evaluate_policy()`：

```
safe       → ✅ 直接执行（读系统状态、查工单等只读操作）
sensitive  → ⚠️ 需要用户确认（代码执行、用户信息查询）
destructive → ❌ 默认禁止，需显式设置 ENABLE_DESTRUCTIVE_TOOLS=true
```

`CodeExecutionTool` 标记为 `tier: "sensitive"`，需要 `required_permissions: ["write"]`：
- 用户无 `write` 权限 → 直接拒绝
- 用户有权限 → 标记 pending，等待确认后执行

```python
# tools/code_execution_tool.py
class CodeExecutionTool(BaseTool):
    tier: str = "sensitive"
    required_permissions: list[str] = ["write"]
```

#### L2 — 静态恶意模式黑名单

执行前 `_is_safe()` 扫描代码是否命中 19 种危险模式，命中任意一条即拒绝：

```python
_DANGEROUS_PATTERNS = [
    # Python
    "import os", "import subprocess", "import sys",
    "os.system", "os.popen", "os.exec",
    "subprocess.call", "subprocess.Popen",
    "__import__", "eval(", "exec(",
    # Node.js
    "require('child_process')", "require('fs')",
    "process.exit", "Deno.run",
    # Shell
    "rm -rf", "mkfs.", "dd if=", "> /dev/",
    "chmod", "chown", "sudo",
]
```

命中 → 返回 `"代码包含不安全操作，已被拒绝执行"`。

#### L3 — 编程语言白名单

```python
# config/settings.py — CodeExecutionSettings
allowed_languages = ["javascript", "typescript", "python", "bash", "arkts"]
```

不在白名单 → 返回 `"不支持的语言 'xxx'，当前支持: ..."`。可通过 `CODE_ALLOWED_LANGUAGES` 环境变量配置。

#### L4 — 环境感知安全策略

```python
# config/settings.py — RuntimeSettings
# development: allow_local_code_execution = True   (允许 subprocess 执行)
# production:  allow_local_code_execution = False  (禁止！必须接入容器沙箱)
```

生产环境自动禁止本地 subprocess 执行，返回：
> "生产环境禁止本机 subprocess 代码执行；请接入 Docker/gVisor/Firecracker 沙箱后再开启。"

这是**防止 dev 宽松策略泄漏到生产**的关键防线。

#### L5 — Subprocess 进程级隔离

实际执行的完整流程：

```
1. 写入临时文件 (tempfile.NamedTemporaryFile, prefix="code_exec_")
        ↓
2. 选择解释器:
   Python  → python3 <.py>          完整执行 + PYTHONUNBUFFERED=1
   JS/TS   → node --check <.js>     语法检查 / node --eval <.ts>
   ArkTS   → node --eval <.ets>     Node 运行时执行
   Bash    → bash -n <.sh>          ⚠️ 仅语法检查，不实际执行！
        ↓
3. asyncio.create_subprocess_exec()  异步子进程
        ↓
4. asyncio.wait_for(timeout=15s)     超时管控
        ↓
5. 超时 → proc.kill() + await proc.wait()  强制终止进程
        ↓
6. os.unlink(tmp_path)              清理临时文件
        ↓
7. 输出截断: stdout/stderr 各最多 2000 字符
```

关键限制与特殊处理：

| 限制项 | 默认值 | 说明 |
|--------|--------|------|
| 超时 | 15 秒 | `CODE_SANDBOX_TIMEOUT`，超时强制 kill |
| 内存 | 256MB | `CODE_SANDBOX_MAX_MEMORY_MB`，需容器层强制实施 |
| 输出截断 | 2000 字符 | 防止大量输出撑爆内存 |
| Bash | `-n` 仅语法检查 | **实际不执行**，最安全的模式 |
| 临时文件 | `delete=False` + finally `os.unlink` | 确保即使异常也清理 |

#### L6 — 熔断器 + 重试 + 审计

`ToolExecutor` 为每个工具维护独立熔断器：

```
连续失败 5 次 → 熔断器打开 → 30 秒内所有请求直接拒绝
         ↓
     冷却 30 秒 → 半开 → 下次成功则关闭
```

重试策略：
- `max_retries: 1` — 代码执行最多重试 1 次
- 重试间隔: `0.1 × attempt` 秒
- 工作流层面：`execute_code` → 失败 → `generate_code` 修复 → 再执行（最多 2 次代码修复重试）

审计日志（PostgreSQL `tool_audit_log` 表）：
- `trace_id` — 关联整条请求链
- `tool_name` / `input_summary` / `output_summary` — 执行摘要
- `success` / `error` / `latency_ms` — 执行结果与耗时

#### 完整执行链路

```
用户查询 "给我一个调用 X API 的代码示例"
        │
        ▼
deep_intent_recognition → code_generation
        │
        ▼
master_agent 调度 → retrieve_knowledge (CodeGenerationWorkflow)
        │
        ▼
generate_code_node → code_agent.generate_code()
        │               └─ LLM 生成代码 / 模板兜底 + AST 符号提取
        ▼
execute_code_node → CodeExecutionTool.execute()
        │
        ├─ ❌ production + !allow_local → 拒绝
        ├─ ❌ 空代码 → 拒绝
        ├─ ❌ L2 黑名单命中 → 拒绝
        ├─ ❌ L3 语言不在白名单 → 拒绝
        ├─ ⚠️ L1 策略门控 → pending 等待用户确认
        │
        ├─ ✅ L5 subprocess 隔离执行 → 15s 超时管控
        │       ├─ 成功(exit_code=0) → 结果注入 generate_answer，标注 "✅ 代码已在沙箱中成功执行"
        │       ├─ 失败 + 未达重试上限 → generate_code 修复代码 → 再执行 (max 2)
        │       └─ 失败 + 耗尽重试 → generate_answer 带免责声明 "⚠️ 代码未经过执行验证，仅供参考"
        │
        └─ L6 熔断器跟踪 → PostgreSQL 审计日志
```

#### 当前局限 & 未来升级路径

| 维度 | 当前实现 | 理想方案 |
|------|---------|---------|
| 进程隔离 | subprocess（同主机） | Docker 容器 / gVisor（内核级沙箱） |
| 内存限制 | 设计 256MB，subprocess 未强制 | cgroups v2 / Firecracker microVM |
| 文件系统 | 共享宿主机 /tmp | tmpfs + chroot / overlayfs |
| 网络隔离 | 无（可访问网络） | 禁用网络 / netns 网络命名空间 |
| Bash 执行 | 仅语法检查 `-n` | Docker 容器内受限执行 |

> CodeExecutionTool docstring: *"Currently uses subprocess with timeout. Future versions can upgrade to Docker/gVisor sandbox for stronger isolation."*

### 配置

```bash
# .env
CODE_SANDBOX_TIMEOUT=15.0
CODE_SANDBOX_MAX_MEMORY_MB=256
MAX_CODE_RETRIES=2
CODE_ALLOWED_LANGUAGES=javascript,typescript,python,bash,arkts
```

---

## 2. Symbol 级代码理解（AST）

### 概述

将 Graph RAG 的实体抽取从纯文本 regex 升级为**代码块感知 + AST 级符号解析**，从文档中的代码块提取精确的代码符号。

### 新增实体类型

| 实体类型 | 示例 | 提取方式 |
|----------|------|----------|
| `IMPORT` | `@ohos.app.ability` | AST / regex |
| `METHOD_CALL` | `console.log()` | AST / regex |
| `PROPERTY` | `ability.context` | AST / regex |
| `TYPE` | `type MyType = ...` | AST / regex |
| `INTERFACE` | `interface MyInterface {}` | AST / regex |
| `CODE_BLOCK` | 代码块标记 | markdown fence 检测 |

### 新增关系类型

| 关系 | 含义 | 示例 |
|------|------|------|
| `IMPORTS` | 模块导入 | `import { X } from 'Y'` |
| `EXTENDS` | 类继承 | `class A extends B` |
| `IMPLEMENTS` | 接口实现 | `class A implements B` |

### AST 解析策略

```
代码块检测 → 语言识别
  ├─ TypeScript/JS/ArkTS → tree-sitter (AST) → 置信度 1.0
  ├─ Python → stdlib ast → 置信度 1.0
  └─ 其他 → 增强 regex → 置信度 0.7
```

---

## 3. 示例代码优先检索

### 概述

检索结果中包含代码块的文档获得额外权重，使开发者查询时优先返回带有可运行示例的文档。

### Boost 公式

```
boosted_score = fused_score × (1.0 + boost_factor × code_density)
```

- `code_density`：代码块字符数 / 总字符数（0.0 ~ 1.0）
- `boost_factor`：默认 0.5（最多 +50%）

---

## 4. 外部知识源补充检索

### 概述

当内部知识库检索结果不理想时，自动从外部知识源补充检索，提升开发者查询的覆盖率。

### 支持的外部源

| 源 | 描述 | API |
|----|------|-----|
| **GitHub Issues** | 仓库 Issues 搜索 | GitHub Search API |
| **Stack Overflow** | 技术问答搜索 | Stack Exchange API |
| **Web Search** | 搜索引擎 | SerpAPI / Bing API |

### 触发条件

已集成到主 `retrieve_knowledge` 节点中，每次检索都会并行尝试外部搜索。每个外部源独立错误处理，一个源失败不影响其他源，也不影响主流程。

---

## 5. Cross-Encoder 精排 🆕

### 概述

使用 Ollama 部署的 `pdurugyan/qwen3-reranker-0.6b-q8_0` 模型进行 Cross-Encoder 精排，替换原有的规则关键词重排序。

Cross-Encoder 将 (query, document) 作为联合输入处理，比 Bi-Encoder 和关键词方法更准确地判断相关性。这是 TECHNICAL_DEEP_DIVE 文档中排名 #1 的改进建议。

### 重排序链

```
Cross-Encoder (Ollama qwen3) → API Reranker → 规则(关键词+来源多样性)
       ↑ 最高精度                  ↑ 可配置端点      ↑ 始终可用兜底
```

### 工作原理

1. 检索结果合并后，取 top-20 候选文档
2. 每个 (query, document) 对送入 qwen3-reranker 独立打分
3. 批次处理（默认 20 条/批），避免超过模型上下文限制
4. 分数归一化后排序，返回 top-N

### 配置

```bash
# .env
RERANKER_ENABLED=true
RERANKER_MODEL=pdurugyan/qwen3-reranker-0.6b-q8_0:latest
OLLAMA_BASE_URL=http://localhost:11434
RERANKER_TIMEOUT=30.0
RERANKER_BATCH_SIZE=20
```

### 安装模型

```bash
ollama pull pdurugyan/qwen3-reranker-0.6b-q8_0:latest
```

### Fallback 保证

Ollama 不可用时自动回退：API Reranker → 规则重排，主流程不受影响。

### 预期收益

- `context_precision` 提升 8%-15%
- 多文档场景下相关性排序更准确

---

## 6. 语义缓存 🆕

### 概述

双层语义缓存，通过 embedding 相似度匹配，让热门问题直接命中缓存，减少 LLM 调用成本和 P95 延迟。这是 TECHNICAL_DEEP_DIVE 文档中排名 #3 的改进建议。

### 缓存层级

```
查询进入
  ├── Layer 1: 精确匹配 (SHA256 哈希)
  │     └── 命中率 ~10%，延迟 <1ms
  │
  ├── Layer 2: 语义匹配 (Embedding Cosine Similarity)
  │     └── 命中率 ~30%，延迟 ~5ms，阈值 ≥ 0.92
  │
  └── Cache Miss → 全量检索流程
```

### 配置

```bash
# .env
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_TTL=3600            # 缓存 TTL（秒），默认 1 小时
SEMANTIC_CACHE_SIMILARITY=0.92     # 语义相似度阈值
SEMANTIC_CACHE_MAX_ENTRIES=1000    # 最大条目数
```

### 存储

- **生产**：Redis（TTL 自动过期）
- **开发**：内存 LRU（OrderedDict，自动淘汰）

### 预期收益

- 热门问题 P95 延迟下降 60%-80%
- LLM Token 成本下降（缓存命中直接返回）
- 缓存命中率随使用量增长

---

## 7. Claim-level 校验 🆕

### 概述

将答案拆分为原子断言（Claim），逐条对照源文档验证是否基于证据。替换原有的整体式校验，实现细粒度幻觉检测。这是 TECHNICAL_DEEP_DIVE 文档中排名 #2 的改进建议。

### 断言类型

| 断言类型 | 检测方式 | 示例 |
|----------|---------|------|
| `api` | API 引用是否在源文档中出现 | `@ohos.app.ability` |
| `code` | 代码标识符是否在源文档中出现 | `createAbility()` |
| `version` | 版本号是否在源文档中确认 | `API 12` |
| `error_code` | 错误码是否在源文档中出现 | `15500000` |
| `comparison` | 对比双方是否都在源文档中 | "A 比 B 更..." |
| `factual` | 关键词重叠率 ≥ 50% | 一般事实陈述 |
| `migration` | 迁移路径是否在源文档中确认 | "从 FA 迁移到 Stage" |

### 工作流

```
草稿答案
  → 句子拆分
  → 断言分类（6种类型）
  → 逐条对照源文档
  → 置信度打分 + 幻觉标注
  → 传统校验兜底
  → (verified, reason, claim_result) 输出
```

### 输出

```json
{
  "total_claims": 5,
  "grounded_claims": 4,
  "hallucinated_claims": 1,
  "hallucination_rate": 0.20,
  "claims": [
    {"text": "API 12 支持...", "type": "version", "grounded": true, "confidence": 0.95},
    {"text": "不存在的 API...", "type": "api", "grounded": false, "confidence": 0.15}
  ]
}
```

### 预期收益

- `hallucination_rate` 下降 20%-40%
- 精确定位哪些断言缺乏证据支撑
- 为 Prompt 优化提供精确的改进方向

---

## 8. 冲突证据检测 🆕

### 概述

当多个检索文档包含矛盾信息时，自动检测并标注冲突，避免答案基于不可靠证据。这是 TECHNICAL_DEEP_DIVE 文档 Context 专题中的改进建议。

### 冲突类型

| 冲突类型 | 检测方式 | 严重度 |
|----------|---------|--------|
| **版本冲突** | 同一 API，不同文档引用不同 API 版本 | `high` |
| **否定冲突** | 一个文档说"不支持"，另一个说"支持" | `high` |
| **API 废弃冲突** | 一个文档使用已废弃的 API | `critical` |
| **建议冲突** | 不同文档推荐不同方案 | `medium` |

### 处理策略

1. 检测到冲突 → 自动标注 `conflict_info` 到文档
2. 冲突警告注入 Prompt → LLM 感知冲突后谨慎回答
3. 高严重度冲突 → 降低该证据权重或提示用户

### 预期收益

- `verification_fail_rate` 下降 10%-20%
- 降低因冲突证据导致错误结论的风险

---

## 9. Prompt Registry（Prompt 版本管理） 🆕

### 概述

Prompt 模板的版本管理、A/B 测试和回滚系统。每次 Prompt 变更可追溯、可对比、可一键回滚。

### 功能

| 功能 | 说明 |
|------|------|
| **版本管理** | 语义化版本（MAJOR.MINOR.PATCH），完整变更历史 |
| **模型感知** | 同一 Prompt 可按模型分别存储不同版本 |
| **A/B 测试** | 流量切分配置（如 v1.0: 80%, v1.1: 20%） |
| **一致性哈希** | 按 user_id 粘性分配，同一用户始终用同一版本 |
| **指标关联** | 每个版本关联质量指标（faithfulness, accuracy 等） |
| **一键回滚** | `registry.rollback("router_prompt", "1.0.0")` |

### 使用示例

```python
from enterprise_agentic_rag.context.prompt_registry import get_prompt_registry

registry = get_prompt_registry()

# 注册新版本
registry.register("router_prompt", "1.1.0", template_str, model="default")

# 获取当前版本
active = registry.get("router_prompt", model="qwen-max")

# A/B 测试
registry.set_traffic_split("router_prompt", {"1.0.0": 0.8, "1.1.0": 0.2})

# 回滚
registry.rollback("router_prompt", "1.0.0")

# 记录指标
registry.record_metrics("router_prompt", "1.1.0", {"accuracy": 0.92})

# 持久化
registry.save()
```

---

## 10. 流式输出（SSE） 🆕

### 概述

`POST /chat/stream` 端点通过 Server-Sent Events 实时推送工作流节点执行事件、深度思考推理链和回答文本增量，前端可展示实时进度和 AI 思考过程。

### 事件类型

```json
{"type": "start", "trace_id": "abc123", "query": "..."}
{"type": "node_end", "node": "retrieve_knowledge", "data": {"node": "retrieve_knowledge", "latency_ms": 85, "success": true}}
{"type": "thinking", "thinking_content": "分析用户问题..."}
{"type": "answer_chunk", "content": "鸿蒙应用开发入门是一个..."}
{"type": "done", "answer": "...", "trace_id": "abc123", "citations": [...], "verified": true, "intent": "concept_qa", "thinking": "完整推理链..."}
{"type": "error", "message": "请求处理超时"}
{"type": "end"}
```

| 事件 | 说明 | 何时发送 |
|------|------|---------|
| `start` | 请求已接收 | 立即 |
| `node_end` | 工作流节点完成 | 每个节点结束时 |
| `thinking` | CoT 推理链片段 | 仅 `deep_thinking: true`，增量推送 |
| `answer_chunk` | 回答文本增量 | 答案生成过程中逐段推送 |
| `done` | 处理完成 | 流程全部结束 |
| `error` | 超时或异常 | 出错时 |
| `end` | 流终止 | 总是最后发送 |

### 使用

```bash
# 开启深度思考
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "鸿蒙应用开发如何入门？", "user_id": "u001", "session_id": "demo", "deep_thinking": true}'

# 关闭深度思考（更快响应）
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "鸿蒙应用开发如何入门？", "user_id": "u001", "session_id": "demo", "deep_thinking": false}'
```

---

## 10b. 深度思考（CoT Visible） 🆕

### 概述

模仿华为智能客服的「深度思考」功能，在生成最终答案前先展示 AI 的推理链（Chain-of-Thought），让用户看到 AI 如何分析问题、如何组织证据、如何规划回答结构。

### 工作原理

```
请求进入 (deep_thinking: true)
  → _generate_thinking_trace() 生成推理链
  ├─ LLM 模式: 调用 LLM 生成 CoT 分析（问题类型→关键信息→结构规划）
  └─ Mock 模式: 关键词识别 + 规则分析
  → SSE 流式推送 thinking 事件
  → 正常答案生成流程
  → SSE 流式推送 answer_chunk 事件
  → done 事件携带完整 thinking 内容
```

### 前端展示

Widget 中的 `DeepThinking` 组件以折叠面板展示推理过程：
- 思考中：金色边框 + "正在思考..." 动画
- 思考完成：可折叠面板 + "思考完成" 标签
- 内容以等宽字体展示，最大高度 300px，超出滚动

### 配置

```bash
# 默认开启，可通过请求参数随时关闭
# 请求体中的 deep_thinking 字段控制
{"query": "...", "deep_thinking": true}   # 开启
{"query": "...", "deep_thinking": false}  # 关闭
```

---

## 11. OpenTelemetry 全链路追踪 🆕

### 概述

替换纯 JSONL 日志为 OTLP 导出的分布式追踪，可在 Jaeger / Tempo 中可视化全链路。这是 TECHNICAL_DEEP_DIVE 文档可观测性专题中的改进建议。

### 配置

```bash
# .env
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=enterprise-agentic-rag
```

### 自动埋点

- FastAPI（所有 HTTP 请求）
- httpx（所有出站 HTTP 调用）
- LangGraph 节点（手动 Span）

### Fallback

OpenTelemetry SDK 未安装或 Collector 不可用时，自动回退到 JSONL 日志。

---

## 12. 自动回滚阈值校准 🆕

### 概述

基于生产指标动态校准回滚阈值，使用 EWMA + 3σ 统计方法，替代静态硬编码阈值。这是 TECHNICAL_DEEP_DIVE 文档 Harness 设计专题中的改进建议。

### 工作原理

```
生产指标流入
  → 滑窗收集（默认 100 点）
  → EWMA 基线计算
  → 标准差分析
  → 动态阈值更新
  → 置信度评估（low/medium/high）
```

### 校准的阈值

| 指标 | 校准方式 | 说明 |
|------|---------|------|
| P95 延迟 Warning | baseline + 3σ 或 2x baseline | 取较大值 |
| P95 延迟 Critical | baseline + 6σ 或 4x baseline | 取较大值 |
| 错误率 Warning | baseline + 3σ，最少 5% | 上限 50% |
| 错误率 Critical | baseline + 5σ，最少 15% | 上限 75% |
| 校验通过率 | baseline - 2σ，最少 60% | 下限保护 |
| 幻觉率 | baseline + 3σ，最少 5% | 上限 50% |

### 回滚决策

至少 2 个阈值同时被突破时才触发回滚建议。

---

## 13. 意图感知检索工作流 🆕

### 概述

`retrieve_knowledge` 节点已升级为根据 Deep Intent 的 `RetrievalMode` 自动分派到 4 个专业检索工作流。

### 工作流分派

| RetrievalMode | 工作流 | 触发意图 | 阶段 |
|---------------|--------|---------|------|
| `hybrid_only` / `parallel` | HybridRAGWorkflow | concept_qa, best_practice, learning_guidance | kw+vec并行→RRF融合→重排→证据选择 |
| `graph_first` | GraphFirstWorkflow | migration, compatibility, architecture | 图谱→扩展→kw+vec并行→三路融合 |
| `error_first` | ErrorFirstWorkflow | error_diagnosis, project_debug | 错误库→FAQ→工单→kw→融合 |
| `code_first` | CodeGenerationWorkflow | code_generation, api_usage | 示例→API参考→官方文档→融合 |

### 5 层回退链

```
语义缓存 → 意图感知工作流 → GraphRAG Orchestrator → 旧版 Retriever → 外部搜索
```

---

## 长期记忆与 LLM 驱动摘要

### LongTermMemory 重要性评分（已升级）

评分从原有的 7 项规则信号升级为 12 项，新增：

| 新增信号 | 加分 | 说明 |
|----------|------|------|
| 语义密度（技术引用数 ≥2） | +0.10 | API/版本/链接/PascalCase 标识符 |
| API 引用密度（≥3 个 @ohos 引用） | +0.10 | 高价值 API 文档内容 |
| 用户反馈加成 | +0.15 | 用户点赞的内容更值得记忆 |
| 时间衰减 | -0.05~0.20 | 超过 1 周降 0.10，超过 30 天降 0.20 |

### 配置参考

```python
from enterprise_agentic_rag.memory import MemoryManager, LongTermMemory

ltm = LongTermMemory(
    importance_threshold=0.6,     # 更严格：只保留高重要性内容
    max_memories_per_user=200,    # 每用户最多 200 条
    dedup_threshold=0.90,         # 更宽松的去重
)

mgr = MemoryManager(long_term=ltm)
```

---

## Eval Gate CI 🆕

### 自动化评估门禁

每次 PR 自动运行 8 条评估用例，强制通过质量门：

```bash
python scripts/run_eval_gate.py --threshold 0.7
```

### 评估维度

| 指标 | 权重 | 说明 |
|------|------|------|
| faithfulness | 40% | 答案忠实度 |
| context_recall | 30% | 上下文召回率 |
| answer_relevancy | 20% | 答案相关性 |
| intent_accuracy | 10% | 意图识别准确率 |

### Agent 决策评估集

22 条测试用例覆盖全部 10 种意图（concept_qa, api_usage, code_generation, error_diagnosis, migration, compatibility, project_debug, best_practice, architecture, learning_guidance），含边界案例和多意图场景。

---

## 配置速查

```bash
# .env 完整配置（新增项）

# Cross-Encoder 精排
RERANKER_ENABLED=true
RERANKER_MODEL=pdurugyan/qwen3-reranker-0.6b-q8_0:latest
OLLAMA_BASE_URL=http://localhost:11434
RERANKER_TIMEOUT=30.0
RERANKER_BATCH_SIZE=20

# 语义缓存
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_TTL=3600
SEMANTIC_CACHE_SIMILARITY=0.92
SEMANTIC_CACHE_MAX_ENTRIES=1000

# OpenTelemetry
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=enterprise-agentic-rag

# 告警阈值
ALERT_P95_LATENCY_WARNING=10.0
ALERT_P95_LATENCY_CRITICAL=30.0
ALERT_ERROR_RATE_WARNING=0.10
ALERT_HALLUCINATION_MAX=0.15

# 回滚校准
ROLLBACK_MIN_DATA_POINTS=50
ROLLBACK_WINDOW_SIZE=100
ROLLBACK_EWMA_ALPHA=0.3
```
