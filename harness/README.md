# Harness — AI Delivery Engineering

> Harness 不负责 Agent 运行时能力本身。  
> Harness 负责 Agent 系统的工程化交付治理。

## 为什么 Agent 系统需要 Harness

传统后端上线关注：单元测试 → 接口可用 → 镜像构建。

Agent 系统上线还必须关注：
- RAG 检索是否命中（hit@k）
- Prompt 是否退化（citation rate）
- 答案是否有引用（groundedness）
- Verifier 是否误放幻觉答案（false_pass_rate）
- Tool Calling 是否稳定（tool_success_rate）
- fallback_rate 是否升高
- thumbs_down_rate 是否升高
- human_fallback_rate 是否异常

因此本项目引入 **Harness-based AI Delivery Pipeline**，把 AI 评估指标作为发布门禁。

## 目录

```
harness/
├── README.md                              # 本文件
├── pipeline/
│   ├── ci.yaml                            # CI: install → test → build → lint → docker
│   ├── cd.yaml                            # CD: dev → smoke → gate → staging → canary → prod
│   ├── eval_gate.yaml                     # Agent Eval Gate: RAG + Answer + Agent 质量门
│   ├── prompt_eval_gate.yaml              # Prompt Eval Gate: intent/route/citation 精度门
│   ├── security_gate.yaml                 # Security Gate: secret/dep/perm/injection/leak
│   └── rollback.yaml                      # 7 维度独立回滚
├── feature_flags/
│   ├── flags.md                           # 20+ Feature Flag 定义
│   └── flag_strategy.md                   # Flag 生命周期 + 治理策略
├── quality_gates/
│   ├── metric_thresholds.md               # 每个指标含义/阈值/处理方案
│   └── quality_gate_policy.md             # 阻断策略 + Override 规则
├── environments/
│   ├── dev.md                             # mock LLM + 快速迭代
│   ├── staging.md                         # 真实 Docker + Eval Gate + Canary
│   └── production.md                      # stable only + 审批 + 监控
├── release/
│   ├── release_checklist.md               # 上线前 checklist
│   ├── canary_strategy.md                 # 灰度策略：分流/放量/停止/回滚
│   └── production_readiness_review.md     # 7 维度生产就绪评审
├── runbooks/
│   ├── deploy_runbook.md                  # 发布到 dev/staging/production
│   ├── rollback_runbook.md                # 7 维度回滚手册
│   ├── eval_gate_runbook.md               # Eval Gate 失败诊断
│   ├── incident_runbook.md                # 5 类故障应急
│   └── feature_flag_runbook.md            # Flag 操作手册
└── templates/
    ├── pipeline_variables.md              # CI/CD 变量定义
    ├── environment_variables.md           # 按环境的变量矩阵
    ├── rollback_plan_template.md          # 回滚计划模板
    └── incident_report_template.md        # 事故报告模板
```

## Harness 十大职责

| # | 职责 | 产出 |
|---|------|------|
| 1 | CI Pipeline | 代码→测试→构建→Lint→镜像 |
| 2 | CD Pipeline | dev→staging→canary→production |
| 3 | Agent Eval Gate | RAG+Answer+Agent 三维度质量门 |
| 4 | Prompt Eval Gate | intent/route/citation 精度门 |
| 5 | Security Gate | secret/dep/perm/injection/leak 扫描 |
| 6 | Feature Flag | 20+ flags, 零停机切换 |
| 7 | Canary Release | 5%→20%→50%→100% 指标驱动 |
| 8 | Environment Promotion | dev→staging→production 晋级规则 |
| 9 | Rollback | Code/Prompt/Workflow/Retriever/LLM/Tool/Verifier |
| 10 | Incident Runbook | 5 类故障诊断 + 止损 SOP |
