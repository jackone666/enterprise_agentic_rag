# Rollback Plan Template

## Release Info
- **Release Version**: `______`
- **Date**: `______`
- **Rollback SLA**: 10 minutes

## Rollback Triggers

| Trigger | Action |
|---------|--------|
| `fallback_rate > 0.30` | Immediate rollback |
| `error_rate > 0.01` | Immediate rollback |
| `verification_pass_rate < 0.70` | Immediate rollback |
| `thumbs_down_rate > 0.15` | Prompt rollback |

## Rollback Commands

```bash
# Code
git checkout <STABLE_TAG> && docker compose up -d

# Prompt
export PROMPT_VERSION=<STABLE_VERSION>

# Retriever
export RETRIEVER_MODE=keyword

# LLM
export LLM_PROVIDER=mock

# Tool
export ENABLE_TOOL_CALLING=false

# Verifier
export VERIFIER_MODE=rule_only
```

## Post-Rollback Verification
- [ ] `/health` → 200
- [ ] `/chat` → returns answer
- [ ] `fallback_rate` back to baseline
- [ ] `pytest` passes

## Rollback Record
- **Trigger**: `______`
- **Start Time**: `______`
- **End Time**: `______`
- **Root Cause**: `______`
