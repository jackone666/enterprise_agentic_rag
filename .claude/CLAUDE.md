# Enterprise Agentic RAG — Claude Code 项目级配置

> 本文件由项目自动加载。每次 Claude Code 启动时生效。

## ⚠️ 强制行为：doc-code-sync 双向同步

**任何时候**修改下列文件，**必须**先调用 `doc-code-sync` skill：

1. `src/enterprise_agentic_rag/**/*.py` —— 改代码 → 同步文档
2. `technical_deep_dive/**/*.md` —— 改文档 → 同步代码验证
3. `config/settings.py` —— 改关键参数 → 同步 CODE_DOC_MAPPING.md §C
4. `recovery/threshold_calibrator.py` / `retry_policy.py` —— 改阈值/重试 → 同步 §C
5. `agents/deep_intent/schema.py` —— 改 Intent/RetrievalMode 枚举 → 同步 §C
6. `graph/state.py` / `workflow.py` —— 改 State 字段或节点 → 同步 §C

### 调用方式

直接使用 Skill 工具调用 `doc-code-sync`，Claude 会自动：
1. 读取 `technical_deep_dive/CODE_DOC_MAPPING.md`
2. 查 §A/§B 找对端章节
3. 检查 §C 关键参数是否要更新
4. 提示需要同步的具体内容
5. 写一行变更到 §E 变更日志

### 用户体验

- 改代码：Claude 会主动说"建议同步到 technical_deep_dive/XX §X.X，是否执行？"
- 改文档：Claude 会主动说"建议验证 src/XX/xx.py 是否仍符合文档描述"
- 关键参数：自动同步 `CODE_DOC_MAPPING.md` §C + 对应文档章节

## 项目约定

- **不**主动 push 任何 commit / PR
- **不**安装/更新 npm/pip 依赖（用户偏好手动控制）
- **优先**使用 ModelScope / hf-mirror 镜像下载
- 中文回复（除非用户切英文）
- 长操作显示进度条（tqdm 风格）
- 工具调用失败时报告真实状态，不掩盖错误

## 关键文件位置

| 内容 | 位置 |
|------|------|
| 文档-代码映射 | `technical_deep_dive/CODE_DOC_MAPPING.md` |
| 项目级 skill 定义 | `.claude/skills/doc-code-sync/SKILL.md` |
| 已有项目 skill | `.claude/skills/{health,run-tests,debug-retrieval,...}/` |
| 架构入口文档 | `technical_deep_dive/01-项目总览与系统架构.md` |
| Agent 体系文档 | `technical_deep_dive/03-Agent-体系与进阶设计.md` |
| RAG 检索文档 | `technical_deep_dive/04-RAG-检索引擎与GraphRAG.md` |
| 主题题库入口 | `technical_deep_dive/主题/README.md` |
| 变更日志 | `technical_deep_dive/CODE_DOC_MAPPING.md` §E |

## 当前 P0 修复状态（截至 2026-06-06）

- ✅ `last_worker` 命名（5 处：context_manager / knowledge_agent / code_agent）
- ✅ `agent/` 合并到 `agents/deep_intent/`
- ✅ `retrieval_router.py` / `__init__.py` 下线
- ✅ AgentState 字段数 67/70 → 72
- ✅ 5 Agent 统一导出
- ✅ 主题文件（01/02）按详细程度扩写 + 流程图重画
- ✅ 03/04/05/06/07 文档加易误会点 + 决策矩阵
- ✅ 置信度/分数/Agent 停止条件写到 03/06/07
- ✅ doc-code-sync skill + 双向映射表
