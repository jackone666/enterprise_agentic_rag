# Deploy Runbook

## 发布到 Dev

```bash
# 1. 合并代码到 develop 分支
git checkout develop && git merge feature/xxx

# 2. 启动本地服务
docker compose up -d postgres redis
python scripts/init_db.py
python scripts/ingest_docs.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --reload --port 8000

# 3. 验证
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"query":"test","user_id":"u001"}'
pytest tests/ -q
```

## 发布到 Staging

```bash
# 1. 确保 CI 通过
#    - pytest 100% pass
#    - npm run build 0 error

# 2. 部署
git checkout main
docker compose up -d
python scripts/init_db.py
python scripts/ingest_docs.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --port 8000 &

# 3. 运行 Eval Gate
python -m enterprise_agentic_rag.evals.regression_eval

# 4. 检查 Gate 结果
#    - hit@k >= 0.80
#    - citation_present_rate >= 0.95
#    - verification_pass_rate >= 0.85
#    - fallback_rate <= 0.25

# 5. 通过 → 标记 ready_for_production
#    不通过 → 回滚 dev，修复后重试
```

## 发布到 Production

```bash
# 1. 前置检查
#    □ Eval Gate 全部通过
#    □ Canary 全部通过
#    □ Release Manager 已审批
#    □ Rollback plan 已就绪

# 2. 执行部署
export PRODUCTION=true
docker compose -f docker-compose.prod.yml up -d
python scripts/init_db.py
.venv/bin/python -m uvicorn enterprise_agentic_rag.app.main:app --port 8000 &

# 3. 部署后验证
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# 4. 监控 30 分钟
#    - fallback_rate 无异常升高
#    - error_rate < 0.01
#    - p95_latency 正常

# 5. 通知
#    slack #deployments: "production deployed ${VERSION}"
```

## 紧急 Hotfix

```bash
# 1. 从 main 创建 hotfix 分支
git checkout main && git checkout -b hotfix/critical-bug

# 2. 修复 + 提交
git commit -m "hotfix: critical bug fix"

# 3. 跳过 Eval Gate（需 Release Manager 审批）
export SKIP_EVAL_GATE=true  # 仅紧急情况

# 4. 直接部署 production
# ... (同上 production 部署步骤)

# 5. 事后补齐 Eval Gate
```
