-- 001_init.sql — Initial schema for Enterprise Agentic RAG
-- Run manually: psql -U rag_user -d enterprise_rag -f 001_init.sql
-- Or via: python scripts/init_db.py

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(128) DEFAULT '',
    role VARCHAR(32) DEFAULT 'basic',
    department VARCHAR(128) DEFAULT '',
    email VARCHAR(256) DEFAULT '',
    permissions TEXT DEFAULT '[]',
    preferred_language VARCHAR(16) DEFAULT 'zh-CN',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(128) UNIQUE NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    summary TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT DEFAULT '',
    intent VARCHAR(64) DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS long_term_memories (
    id SERIAL PRIMARY KEY,
    memory_id VARCHAR(64) UNIQUE NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    content TEXT DEFAULT '',
    importance DOUBLE PRECISION DEFAULT 0.0,
    memory_type VARCHAR(32) DEFAULT 'episodic',
    source_session VARCHAR(128) DEFAULT '',
    source_turn INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    accessed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qa_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(128) DEFAULT '',
    user_id VARCHAR(64) DEFAULT '',
    query TEXT DEFAULT '',
    answer TEXT DEFAULT '',
    intent VARCHAR(64) DEFAULT '',
    citations TEXT DEFAULT '[]',
    verified BOOLEAN DEFAULT TRUE,
    need_human BOOLEAN DEFAULT FALSE,
    fallback_reason VARCHAR(256) DEFAULT '',
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(128) DEFAULT '',
    user_id VARCHAR(64) DEFAULT '',
    tool_name VARCHAR(128) DEFAULT '',
    input_summary TEXT DEFAULT '',
    output_summary TEXT DEFAULT '',
    success BOOLEAN DEFAULT TRUE,
    error TEXT DEFAULT '',
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(128) DEFAULT '',
    user_id VARCHAR(64) DEFAULT '',
    thumbs_up BOOLEAN DEFAULT TRUE,
    feedback_text TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id SERIAL PRIMARY KEY,
    query TEXT DEFAULT '',
    expected_intent VARCHAR(64) DEFAULT '',
    expected_sources TEXT DEFAULT '[]',
    expected_answer_keywords TEXT DEFAULT '[]',
    difficulty VARCHAR(32) DEFAULT 'medium',
    prompt_version VARCHAR(16) DEFAULT 'v1',
    source VARCHAR(32) DEFAULT 'manual',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS failed_cases (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(64) DEFAULT '',
    session_id VARCHAR(128) DEFAULT '',
    query TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    source VARCHAR(32) DEFAULT 'auto',
    payload TEXT DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
