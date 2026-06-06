# Feature Flag Runbook

## 如何开启 Flag

```bash
# 环境变量方式（即时生效，无需重启）
export FLAG_NAME=true

# 或通过 .env 文件（需重启）
echo "FLAG_NAME=true" >> .env
```

## 如何关闭 Flag

```bash
export FLAG_NAME=false
```

## Flag 修改后验证

```bash
# 1. Smoke test
curl http://localhost:8000/health

# 2. 发送测试问题
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"测试问题","user_id":"u001","session_id":"flag_test"}'

# 3. 观察指标
curl http://localhost:8000/metrics
```

## Production Flag 修改流程

1. 提交 Flag Change Request
2. Engineering Lead 审批（高风险 flag 需要 Security Lead 双签）
3. 先在 staging 验证
4. 记录当前 baseline 指标
5. 执行 flag 变更
6. 观察 15 分钟
7. 对比 baseline — 无异常 → 完成；有异常 → 回滚

## 紧急 Flag 回滚

```bash
# 立即回滚到已知安全状态
export enable_real_llm=false           # 回退 mock
export enable_tool_calling=false       # 关闭工具
export enable_answer_verifier=true     # 确保校验开启
export enable_human_fallback=true      # 确保兜底开启
```
