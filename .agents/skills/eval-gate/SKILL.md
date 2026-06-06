---
name: eval-gate
description: Use when running the RAG evaluation quality gate — checking faithfulness, context recall, answer relevancy, and intent accuracy against the built-in eval dataset. Use when the user wants to evaluate system quality, check if changes degraded performance, or verify eval gate pass/fail status.
---

# Eval Gate

## Overview

运行 RAG 评估门禁，基于 8 个内置测试用例检查系统质量。计算 faithfulness（忠实度）、context_recall（上下文召回）、answer_relevancy（答案相关性）、intent_accuracy（意图准确度）四个维度。

## 评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| Faithfulness | 40% | 回答是否基于检索到的证据 |
| Context Recall | 30% | 检索是否覆盖了回答问题所需的信息 |
| Answer Relevancy | 20% | 回答是否与问题相关 |
| Intent Accuracy | 10% | 意图分类是否正确 |

## Implementation

### 运行评估门禁

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 默认阈值 0.7
uv run python scripts/run_eval_gate.py --threshold 0.7 --output eval_results.json
```

### 自定义阈值和输出

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 更严格的门禁
uv run python scripts/run_eval_gate.py --threshold 0.8 --output eval_results_strict.json
```

### 运行后分析结果

```bash
cd /Users/zing/Desktop/开源项目/enterprise_agentic_rag

# 查看结果摘要
python3 -c "
import json
with open('eval_results.json') as f:
    r = json.load(f)
print(f\"Gate: {'✅ PASSED' if r['gate_passed'] else '❌ FAILED'}\")
print(f\"Overall: {r['summary']['overall']}\")
print(f\"Faithfulness: {r['summary']['faithfulness']}\")
print(f\"Context Recall: {r['summary']['context_recall']}\")
print(f\"Answer Relevancy: {r['summary']['answer_relevancy']}\")
print(f\"Intent Accuracy: {r['summary']['intent_accuracy']}\")
print()
print('Per-case:')
for c in r['cases']:
    intent_match = '✅' if c.get('intent_correct') else '❌'
    print(f\"  {intent_match} {c['query'][:50]}...\")
    print(f\"     expected={c['expected_intent']} detected={c.get('detected_intent','?')}\")
    if c.get('errors'):
        for e in c['errors']:
            print(f\"     ⚠️  {e}\")
"
```

## 执行策略

1. **运行前确认后端服务可用**（评估脚本会调用 `Retriever` 和意图分类器）
2. **评估是离线模拟**（当前的 `_proxy_faithfulness` 和 `_proxy_answer_relevancy` 是基于规则的近似，不是 LLM Judge）
3. **Gate 通过不代表生产质量完美**，它只是一个基本质量门槛
4. **CI 中已有相同的 eval gate**（`.github/workflows/eval-gate.yml`），本地运行可以预先发现问题
5. **如果 gate 不通过**，分析每个 case 的得分，重点关注 faithfulness < 0.5 或 intent_correct = False 的 case

## 8 个内置 Evalu Case

| # | Query | Expected Intent |
|---|-------|----------------|
| 1 | HarmonyOS NEXT 如何申请权限？ | api_usage |
| 2 | @ohos.app.ability 迁移到 API 12 有什么变化？ | migration |
| 3 | HarmonyOS 应用中实现 ArkUI 导航的最佳实践是什么？ | best_practice |
| 4 | 错误码 15500000 是什么问题？ | error_diagnosis |
| 5 | 如何用 TypeScript 实现 HarmonyOS 的网络请求？ | code_generation |
| 6 | Stage 模型和 FA 模型的区别是什么？ | concept_qa |
| 7 | HarmonyOS NEXT API 12 兼容性需要注意什么？ | compatibility |
| 8 | 我的应用在 API 12 上启动崩溃怎么调试？ | project_debug |

## 常见问题

| 问题 | 解决 |
|------|------|
| 评估结果全是 0 | 检查后端服务是否运行，Retriever 是否可用 |
| Intent accuracy 很低 | 检查 `enterprise_agentic_rag.agent.deep_intent.rules` 模块 |
| eval_results.json 为空 | 确认 `--output` 路径有写入权限 |
| CI 和本地结果不一致 | CI 仅启动 PG + Redis，本地有完整服务栈，结果应有差异 |
