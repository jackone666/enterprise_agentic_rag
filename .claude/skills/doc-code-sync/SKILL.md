---
name: doc-code-sync
description: 文档-代码双向同步。当用户修改 src/ 下的代码或 technical_deep_dive/ 下的文档时，自动根据 CODE_DOC_MAPPING.md 检查对应章节是否需要同步，并提示/执行修改。Use this skill whenever the user modifies any file under src/enterprise_agentic_rag/ or technical_deep_dive/ — proactively check if the counterpart needs updating, suggest the specific section, and either auto-apply or ask for confirmation.
---

# doc-code-sync：文档-代码双向同步

> **核心原则**：改代码必须同步文档，改文档必须同步代码。**任何修改都自动检查对端**。

## 触发场景

| 用户动作 | 触发模式 |
|---------|---------|
| 修改 `src/enterprise_agentic_rag/**/*.py` | 反向：找文档 → 提示/同步 |
| 修改 `technical_deep_dive/**/*.md` | 反向：找代码 → 提示/验证 |
| 修改 `config/settings.py` 关键参数 | 同步 §C 关键参数表 + 文档 |
| 修改 `recovery/threshold_calibrator.py` | 同步 §C + 07-可观测性 §10.7 |
| 修改 `recovery/retry_policy.py` | 同步 §C + 06-工具安全 §9.4 |
| 修改 `agents/deep_intent/schema.py` Intent/RetrievalMode | 同步 §C + 03-Agent §4.2.1 |
| 修改 `graph/state.py` 字段 | 同步 §C + 01-项目总览 §1.1 |
| 修改 `graph/workflow.py` 节点/路由 | 同步 §C + 02-LangGraph |

## 工作流

```
[用户修改文件 X]
  │
  ├─> 读取 technical_deep_dive/CODE_DOC_MAPPING.md
  │     ├─> §A. 代码 → 文档 映射
  │     ├─> §B. 文档 → 代码 反向映射
  │     └─> §C. 关键参数速查
  │
  ├─> 定位 X 在 §A/§B 中的条目
  │     └─> 拿到对应文档章节 Y
  │
  ├─> [双向检查 1] 改动是否影响 §C 关键参数？
  │     ├─> 是 → 同步更新 §C + 对应文档
  │     └─> 否 → 继续
  │
  ├─> [双向检查 2] 改动是否影响文档描述的行为？
  │     ├─> 是 → 提示用户："文档 §X 描述与新代码不一致，建议更新为：..."
  │     │     ├─> 用户确认 → 自动更新
  │     │     └─> 用户拒绝 → 记入 changelog
  │     └─> 否 → 继续
  │
  └─> [双向检查 3] 文档是否描述了代码里没有的接口？
        ├─> 是 → 提示用户删除或补实现
        └─> 否 → 结束
```

## 自动执行清单（每次都跑）

1. **代码改动后必跑**：
   - [ ] 查 `CODE_DOC_MAPPING.md` §A 找对应文档章节
   - [ ] 比对 `git diff` 与文档描述
   - [ ] 检查 §C 关键参数是否需要更新
   - [ ] 在 `CODE_DOC_MAPPING.md` §E 写一行变更记录

2. **文档改动后必跑**：
   - [ ] 查 `CODE_DOC_MAPPING.md` §B 找对应代码文件
   - [ ] grep 验证代码仍包含文档提到的关键函数/类/常量
   - [ ] 检查 §C 关键参数是否需要更新
   - [ ] 在 `CODE_DOC_MAPPING.md` §E 写一行变更记录

## 关键参数同步规则

§C 列出的 25+ 关键参数是**最容易脱节的**。任何修改都要走这个流程：

```python
# 检测逻辑（伪代码）
if file_contains_any([
    "MAX_GRAPH_STEPS", "max_graph_steps", "step_count",
    "chunk_size", "top_k", "RRF", "k=",
    "TokenBudget", "max_tokens",
    "confidence", "RETRY", "max_retries",
    "p95_latency", "error_rate", "hallucination",
    "IntentCategory", "RetrievalMode",
    "5 个 Agent", "5 Agent", "16 节点", "16 nodes",
    "72 字段", "72 fields",
]):
    return "TRIGGER_FULL_SYNC"
```

## 输出格式

每次触发 skill 时，输出**三段**：

```
=== doc-code-sync 触发 ===

【改动的文件】X

【对应文档/代码】Y
  - 章节：technical_deep_dive/03-Agent §4.2.2
  - 关键参数：max_graph_steps=18

【建议同步】
  1. 文档 §4.2.2.C 中 max_graph_steps 数值需更新
  2. CODE_DOC_MAPPING.md §C 需更新

【是否执行？】(y/n)
```

## 重要：自动激活方式

skill 通过 `Skill` 工具调用触发。**项目启动时自动加载**的方式：
- 在 `.claude/CLAUDE.md`（项目级）加指令："任何时候改 src/ 或 technical_deep_dive/ 下的文件，必须先调用 doc-code-sync skill"

详见 `.claude/CLAUDE.md` 配置。

## 失败兜底

如果 `CODE_DOC_MAPPING.md` 不存在或被破坏：
1. 重新生成（基于 src/ 目录结构 + technical_deep_dive/ 章节标题做自动推断）
2. 在 §E 写 "REBUILT_MAPPING" 标记

## 维护

- **不要**修改本 SKILL.md 的 frontmatter（`name` / `description`）—— 改了 Claude Code 就识别不到
- 可以**添加**工作流细节到正文
- `CODE_DOC_MAPPING.md` 是数据文件，可以自由更新
