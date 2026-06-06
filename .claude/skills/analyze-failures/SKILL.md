---
name: analyze-failures
description: Use when the user wants to export and analyze failed RAG cases — identify failure patterns, common error types, and suggest improvements. Use after running the eval gate, or when investigating production quality issues.
---

# Analyze Failures

## Overview

导出失败案例并进行分析：从 PostgreSQL 导出 → 归类失败模式 → 输出改进建议。

## 流程

```
export_failed_cases.py → data/eval/exported_failed_cases.jsonl → 
归类分析（意图错误 / 检索不足 / 幻觉 / 权限问题） → 改进建议
```

## Implementation

### 步骤 1: 导出失败案例

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 从 PostgreSQL 导出
uv run python scripts/export_failed_cases.py

# 或指定输出路径
uv run python scripts/export_failed_cases.py data/eval/my_failed_cases.jsonl
```

### 步骤 2: 分析失败模式

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 快速统计
python3 -c "
import json
from collections import Counter

try:
    with open('data/eval/exported_failed_cases.jsonl') as f:
        cases = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    # 尝试 JSONL fallback
    try:
        with open('data/eval/failed_cases.jsonl') as f:
            cases = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print('No failed cases found. Sources:')
        print('  1. PostgreSQL (run export_failed_cases.py)')
        print('  2. data/eval/failed_cases.jsonl (JSONL fallback)')
        exit(1)

print(f'Total failed cases: {len(cases)}')
print()

# 分析意图分类错误
intent_errors = [c for c in cases if c.get('error_type') == 'intent_misclassify' 
                 or 'intent' in str(c.get('errors', '')).lower()]
print(f'Intent classification errors: {len(intent_errors)}')

# 分析检索不足
retrieval_errors = [c for c in cases if c.get('error_type') == 'low_recall'
                    or 'retriev' in str(c.get('errors', '')).lower()]
print(f'Retrieval/recall errors: {len(retrieval_errors)}')

# 分析幻觉
hallucination = [c for c in cases if c.get('error_type') == 'hallucination'
                 or 'faithful' in str(c.get('errors', '')).lower()]
print(f'Faithfulness/hallucination errors: {len(hallucination)}')

# 列出所有错误类型
error_types = Counter()
for c in cases:
    et = c.get('error_type', 'unknown')
    error_types[et] += 1
print()
print('Error type distribution:')
for et, count in error_types.most_common():
    print(f'  {et}: {count}')

# 显示最近 5 条
print()
print('Most recent 5 failed cases:')
for c in cases[:5]:
    print(f\"  [{c.get('created_at','?')}] {c.get('query','?')[:60]}\")
    print(f\"    type={c.get('error_type','?')} errors={c.get('errors','?')}\")
"
```

### 步骤 3: 生成改进建议

基于分析结果，重点关注以下改进方向：

| 失败模式 | 可能原因 | 改进方向 |
|----------|---------|---------|
| 意图分类持续错误 | 规则覆盖不足 | 补充 `deep_intent/rules.py` 关键词规则 |
| 检索召回低 | 文档缺失或索引不足 | 检查 `data/docs/` 文档完整性，增加 BM25 权重 |
| 忠实度低（幻觉） | LLM 未严格遵循证据 | 调整 Verifier agent 的 claim_verification 阈值 |
| 特定领域持续失败 | 领域文档缺失 | 补充对应领域的 markdown 文档 |
| 权限过滤过度 | ACL 配置过严 | 检查用户 permissions 和文档 ACL 匹配 |

## 数据源

失败案例来自两个地方：

1. **PostgreSQL** `failed_cases` 表 — 生产/评估运行中产生的失败记录
2. **`data/eval/failed_cases.jsonl`** — JSONL 格式的本地 fallback

如果两者都没有数据，说明还没有运行过评估门禁或者没有失败案例。

## 常见问题

| 问题 | 解决 |
|------|------|
| PostgreSQL 不可用 | 检查 `data/eval/failed_cases.jsonl` 本地文件 |
| 导出 0 条记录 | 还没有失败案例，先运行 `eval-gate` skill |
| 分析结果不准确 | 当前分析基于规则匹配，生产环境建议接入 RAGAS |
