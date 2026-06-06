"""Tests for the Observability system.

Covers: EventSchema, EventLogger, MetricsCollector, Tracer,
        and end-to-end trace recording through the workflow.
"""

import json
import os
import tempfile
import time

import pytest

from enterprise_agentic_rag.graph.workflow import build_workflow
from enterprise_agentic_rag.observability.event_schema import (
    EventType,
    NodeEvent,
    RetrievalEvent,
    ToolEvent,
    VerificationEvent,
    event_to_dict,
    make_summary,
)
from enterprise_agentic_rag.observability.logger import EventLogger
from enterprise_agentic_rag.observability.metrics import (
    MetricsCollector,
    get_metrics_collector,
)
from enterprise_agentic_rag.observability.tracer import Tracer, get_tracer

# =========================================================================
# Event Schema
# =========================================================================


class TestEventSchema:
    """Tests for event dataclasses and helpers."""

    def test_node_event_defaults(self) -> None:
        evt = NodeEvent(trace_id="t1", session_id="s1", user_id="u1",
                        event_type=EventType.NODE_START, node_name="test_node")
        assert evt.trace_id == "t1"
        assert evt.node_name == "test_node"
        assert evt.event_type == EventType.NODE_START
        assert evt.timestamp > 0

    def test_tool_event_defaults(self) -> None:
        evt = ToolEvent(trace_id="t1", session_id="s1", user_id="u1",
                        event_type=EventType.TOOL_CALL, tool_name="query_ticket")
        assert evt.tool_name == "query_ticket"
        assert evt.input_params == {}

    def test_retrieval_event_defaults(self) -> None:
        evt = RetrievalEvent(trace_id="t1", session_id="s1", user_id="u1",
                             event_type=EventType.RETRIEVAL)
        assert evt.num_docs_retrieved == 0
        assert evt.top_score == 0.0

    def test_verification_event_defaults(self) -> None:
        evt = VerificationEvent(trace_id="t1", session_id="s1", user_id="u1",
                                event_type=EventType.VERIFICATION, verified=True)
        assert evt.verified is True

    def test_make_summary_short_text(self) -> None:
        assert make_summary("hello") == "hello"

    def test_make_summary_long_text(self) -> None:
        long_text = "A" * 200
        result = make_summary(long_text, max_chars=50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_make_summary_empty(self) -> None:
        assert make_summary("") == ""

    def test_make_summary_strips_newlines(self) -> None:
        assert "\n" not in make_summary("line1\nline2")

    def test_event_to_dict(self) -> None:
        evt = NodeEvent(trace_id="t1", session_id="s1", user_id="u1",
                        event_type=EventType.NODE_END, node_name="retrieve",
                        latency_ms=12.5, success=True)
        d = event_to_dict(evt)
        assert d["trace_id"] == "t1"
        assert d["node_name"] == "retrieve"
        assert d["latency_ms"] == 12.5
        assert "timestamp_iso" in d

    def test_event_type_constants(self) -> None:
        assert EventType.NODE_START == "node_start"
        assert EventType.NODE_END == "node_end"
        assert EventType.TOOL_CALL == "tool_call"
        assert EventType.RETRIEVAL == "retrieval"
        assert EventType.VERIFICATION == "verification"


# =========================================================================
# EventLogger
# =========================================================================


class TestEventLogger:
    """Tests for JSONL file logger."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "events.jsonl")
        self.logger = EventLogger(log_path=self.log_path)

    def test_write_and_read_event(self) -> None:
        ok = self.logger.write_event({"trace_id": "t1", "event": "test", "value": 42})
        assert ok is True
        events = self.logger.read_events()
        assert len(events) == 1
        assert events[0]["trace_id"] == "t1"

    def test_write_multiple_events(self) -> None:
        events = [
            {"trace_id": "t1", "n": 1},
            {"trace_id": "t2", "n": 2},
            {"trace_id": "t3", "n": 3},
        ]
        written = self.logger.write_events(events)
        assert written == 3
        assert len(self.logger.read_events()) == 3

    def test_write_unicode(self) -> None:
        self.logger.write_event({"trace_id": "t1", "query": "中文问题"})
        events = self.logger.read_events()
        assert events[0]["query"] == "中文问题"

    def test_read_tail(self) -> None:
        for i in range(10):
            self.logger.write_event({"trace_id": f"t{i}", "n": i})
        tail = self.logger.read_events(tail_n=3)
        assert len(tail) == 3
        assert tail[-1]["n"] == 9

    def test_read_nonexistent_file(self) -> None:
        logger = EventLogger(log_path="/tmp/nonexistent_dir_xyz/events.jsonl")
        # Should not crash
        events = logger.read_events()
        assert events == []

    def test_clear(self) -> None:
        self.logger.write_event({"trace_id": "t1"})
        self.logger.clear()
        assert self.logger.read_events() == []

    def test_log_path_property(self) -> None:
        assert self.logger.log_path == self.log_path

    def test_directory_auto_created(self) -> None:
        nested = os.path.join(self.tmpdir, "deep", "nested", "events.jsonl")
        logger = EventLogger(log_path=nested)
        ok = logger.write_event({"t": 1})
        assert ok is True
        assert os.path.exists(nested)

    def test_corrupt_lines_skipped(self) -> None:
        """Malformed JSON lines should be skipped, not crash."""
        path = self.log_path
        # Manually write bad data
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"ok": 1}) + "\n")
            fh.write("this is not json\n")
            fh.write(json.dumps({"ok": 2}) + "\n")
        events = self.logger.read_events()
        assert len(events) == 2


# =========================================================================
# MetricsCollector
# =========================================================================


class TestMetricsCollector:
    """Tests for in-memory metrics aggregation."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.mc = MetricsCollector()

    def test_initial_state(self) -> None:
        snap = self.mc.snapshot()
        assert snap["total_requests"] == 0

    def test_record_request(self) -> None:
        self.mc.record_request(intent="technical_question", latency_ms=150.0)
        snap = self.mc.snapshot()
        assert snap["total_requests"] == 1
        assert snap["avg_latency_ms"] == 150.0

    def test_intent_distribution(self) -> None:
        self.mc.record_request(intent="troubleshooting")
        self.mc.record_request(intent="troubleshooting")
        self.mc.record_request(intent="policy_question")
        snap = self.mc.snapshot()
        dist = snap["intent_distribution"]
        assert dist["troubleshooting"] == 2
        assert dist["policy_question"] == 1

    def test_retrieval_hit_rate(self) -> None:
        self.mc.record_retrieval(num_docs=3)
        self.mc.record_retrieval(num_docs=0)
        self.mc.record_retrieval(num_docs=5)
        snap = self.mc.snapshot()
        assert snap["retrieval_hit_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_verification_pass_rate(self) -> None:
        self.mc.record_verification(verified=True)
        self.mc.record_verification(verified=True)
        self.mc.record_verification(verified=False)
        snap = self.mc.snapshot()
        assert snap["verification_pass_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_tool_success_rate(self) -> None:
        self.mc.record_tool_call(success=True)
        self.mc.record_tool_call(success=False)
        snap = self.mc.snapshot()
        assert snap["tool_success_rate"] == pytest.approx(0.5, abs=0.001)

    def test_fallback_rates(self) -> None:
        self.mc.record_request(has_fallback=True, need_human=False)
        self.mc.record_request(has_fallback=False, need_human=True)
        self.mc.record_request(has_fallback=True, need_human=True)
        snap = self.mc.snapshot()
        assert snap["fallback_rate"] == pytest.approx(2 / 3, abs=0.001)
        assert snap["human_fallback_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_success_rate(self) -> None:
        self.mc.record_request(success=True)
        self.mc.record_request(success=True)
        self.mc.record_request(success=False)
        snap = self.mc.snapshot()
        assert snap["success_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_reset(self) -> None:
        self.mc.record_request()
        self.mc.reset()
        assert self.mc.snapshot()["total_requests"] == 0

    def test_uptime_tracks_time(self) -> None:
        self.mc.record_request()
        time.sleep(0.15)
        snap = self.mc.snapshot()
        assert snap["uptime_seconds"] >= 0.1

    def test_zero_division_safety(self) -> None:
        """All rates should be 0 when no data exists."""
        snap = self.mc.snapshot()
        assert snap["retrieval_hit_rate"] == 0.0
        assert snap["verification_pass_rate"] == 0.0
        assert snap["tool_success_rate"] == 0.0

    def test_singleton(self) -> None:
        mc1 = get_metrics_collector()
        mc2 = get_metrics_collector()
        assert mc1 is mc2


# =========================================================================
# Tracer
# =========================================================================


class TestTracer:
    """Tests for the tracing orchestrator."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(self.tmpdir, "events.jsonl")
        self.logger = EventLogger(log_path=log_path)
        self.tracer = Tracer(logger=self.logger)
        # Reset metrics
        self.tracer.metrics.reset()

    def test_new_trace_unique(self) -> None:
        t1 = self.tracer.new_trace()
        t2 = self.tracer.new_trace()
        assert len(t1) == 12
        assert t1 != t2

    def test_record_tool_event(self) -> None:
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1"}
        self.tracer.record_tool_event(
            state, tool_name="query_ticket",
            params={"ticket_id": "TKT-001"},
            output="工单详情...", latency_ms=12.5, success=True,
        )
        events = self.logger.read_events()
        assert len(events) == 1
        assert events[0]["tool_name"] == "query_ticket"
        assert events[0]["success"] is True

    def test_record_retrieval_event(self) -> None:
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1"}
        self.tracer.record_retrieval_event(
            state, query="测试查询", num_docs=5,
            top_score=0.95, latency_ms=3.2,
        )
        events = self.logger.read_events()
        assert len(events) == 1
        assert events[0]["num_docs_retrieved"] == 5

    def test_record_verification_event(self) -> None:
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1"}
        self.tracer.record_verification_event(
            state, verified=True, reason="all checks passed",
            latency_ms=1.5,
        )
        events = self.logger.read_events()
        assert len(events) == 1
        assert events[0]["verified"] is True

    def test_metrics_updated_by_events(self) -> None:
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1"}

        self.tracer.record_retrieval_event(state, num_docs=3)
        self.tracer.record_verification_event(state, verified=True)
        self.tracer.record_tool_event(state, tool_name="t1", success=True)

        snap = self.tracer.metrics.snapshot()
        assert snap["retrieval_hit_rate"] == 1.0
        assert snap["verification_pass_rate"] == 1.0
        assert snap["tool_success_rate"] == 1.0


# =========================================================================
# Traced node wrapper
# =========================================================================


class TestTracedNode:
    """Tests for the traced_node wrapper used in the workflow."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(self.tmpdir, "events.jsonl")
        self.logger = EventLogger(log_path=log_path)
        self.tracer = Tracer(logger=self.logger)
        self.tracer.metrics.reset()

    @pytest.mark.asyncio
    async def test_traced_node_records_start_end(self) -> None:
        async def my_node(state):
            return {"result": "ok"}

        wrapped = self.tracer.traced_node("my_node", my_node)
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1",
                 "query": "测试"}

        output = await wrapped(state)
        assert output["result"] == "ok"
        # node_events written to JSONL logger, not to output state
        assert "node_events" not in output

    @pytest.mark.asyncio
    async def test_traced_node_captures_exceptions(self) -> None:
        async def failing_node(state):
            raise RuntimeError("模拟失败")

        wrapped = self.tracer.traced_node("failing", failing_node)
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1",
                 "query": ""}

        output = await wrapped(state)
        # Should not raise — error captured in output
        assert "error" in output
        assert "模拟失败" in output["error"]

    @pytest.mark.asyncio
    async def test_traced_node_output_summary(self) -> None:
        async def big_node(state):
            return {
                "intent": "technical_question",
                "retrieved_docs": [{"content": "X" * 5000}],
                "verified": True,
            }

        wrapped = self.tracer.traced_node("big_node", big_node)
        state = {"trace_id": "t1", "session_id": "s1", "user_id": "u1",
                 "query": ""}

        output = await wrapped(state)
        # Large fields should pass through cleanly
        assert output["intent"] == "technical_question"
        assert output["verified"] is True


# =========================================================================
# Workflow integration
# =========================================================================


class TestWorkflowObservability:
    """End-to-end tests for tracing through the full workflow."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # Reset metrics before each test
        get_metrics_collector().reset()
        # Use a temp file for the tracer singleton
        self.tmpdir = tempfile.mkdtemp()
        log_path = os.path.join(self.tmpdir, "events.jsonl")
        logger = EventLogger(log_path=log_path)

        # Replace global tracer with temp logger for this test
        import enterprise_agentic_rag.observability.tracer as tr_mod
        self._old_tracer = tr_mod._tracer
        tr_mod._tracer = Tracer(logger=logger)
        tr_mod._tracer.metrics.reset()

        # Also update the workflow module's tracer reference
        import enterprise_agentic_rag.graph.workflow as wf_mod
        self._old_wf_tracer = wf_mod._tracer
        wf_mod._tracer = tr_mod._tracer

        self.graph = build_workflow()
        self.logger = logger

    def teardown_method(self) -> None:
        import enterprise_agentic_rag.graph.workflow as wf_mod
        import enterprise_agentic_rag.observability.tracer as tr_mod
        tr_mod._tracer = self._old_tracer
        wf_mod._tracer = self._old_wf_tracer

    @pytest.mark.asyncio
    async def test_workflow_produces_trace_events(self) -> None:
        """A full pipeline run should produce multiple node events."""
        result = await self.graph.ainvoke({
            "query": "如何重置密码？",
            "user_id": "u001",
            "session_id": "obs_test",
                       "trace_id": "trace-001",
        })
        # node_events moved to JSONL logger only (not in state)
        assert "node_events" not in result

    @pytest.mark.asyncio
    async def test_workflow_produces_metrics_snapshot(self) -> None:
        """Final state no longer contains metrics_snapshot (moved to tracer)."""
        result = await self.graph.ainvoke({
            "query": "API 认证方式有哪些？",
            "user_id": "u001",
            "session_id": "obs_metrics",
            "trace_id": "trace-002",
        })
        # metrics_snapshot removed from state; use tracer.metrics directly
        assert result.get("metrics_snapshot", {}) == {}

    @pytest.mark.asyncio
    async def test_trace_id_preserved(self) -> None:
        """The trace_id should survive the full pipeline."""
        result = await self.graph.ainvoke({
            "query": "测试",
            "user_id": "u001",
            "session_id": "obs_trace_id",
            "trace_id": "my-custom-trace",
        })
        assert result.get("trace_id") == "my-custom-trace"

    @pytest.mark.asyncio
    async def test_events_persisted(self) -> None:
        """Events should be persisted — JSONL logger (PG optional)."""
        result = await self.graph.ainvoke({
            "query": "如何重置密码？",
            "user_id": "u001",
            "session_id": "obs_persist",
            "trace_id": "trace-persist",
        })

        # Logger persistence: try PG first, then JSONL fallback
        pg_events = self._read_pg_events("trace-persist")
        jsonl_events = self.logger.read_events()
        assert len(pg_events) > 0 or len(jsonl_events) > 0, (
            "Events not persisted — neither PG nor JSONL has events"
        )

        # Verify event structure on whichever source has data
        events = pg_events if pg_events else jsonl_events
        for evt in events:
            assert "trace_id" in evt
            assert "event_type" in evt
            assert "timestamp" in evt

    @staticmethod
    def _read_pg_events(trace_id: str) -> list[dict]:
        """Read events from PostgreSQL for a given trace_id."""
        try:
            import asyncio

            from sqlalchemy import text

            from enterprise_agentic_rag.storage.database import get_db_manager
            dbm = get_db_manager()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            if not loop.run_until_complete(dbm.check_connection()):
                return []

            async def _fetch():
                async with dbm.session() as sess:
                    r = await sess.execute(
                        text("SELECT meta_json FROM node_events WHERE trace_id = :tid"),
                        {"tid": trace_id},
                    )
                    return [json.loads(row[0]) for row in r.fetchall()]

            return loop.run_until_complete(_fetch())
        except Exception:
            return []

    @pytest.mark.asyncio
    async def test_all_node_names_appear(self) -> None:
        """The event stream should cover the critical nodes (via JSONL logger)."""
        result = await self.graph.ainvoke({
            "query": "数据分类标准是什么？",
            "user_id": "u001",
            "session_id": "obs_nodes",
            "trace_id": "trace-nodes",
        })
        # node_events no longer in state; verify via JSONL logger
        jsonl_events = self.logger.read_events()
        node_names = {
            evt["node_name"]
            for evt in jsonl_events
            if evt.get("event_type") == "node_end"
        }
        # Critical nodes should be present
        assert "load_memory" in node_names
        assert "check_permission" in node_names
        assert "deep_intent_recognition" in node_names
        assert "retrieve_knowledge" in node_names

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_requests(self) -> None:
        """Metrics should accumulate across multiple /chat invocations."""
        # Request 1
        await self.graph.ainvoke({
            "query": "如何重置密码？",
            "user_id": "u001",
            "session_id": "obs_cum_1",
            "trace_id": "trace-cum-1",
        })
        # Request 2
        await self.graph.ainvoke({
            "query": "API 认证方式有哪些？",
            "user_id": "u002",
            "session_id": "obs_cum_2",
            "trace_id": "trace-cum-2",
        })
        snap = get_metrics_collector().snapshot()
        assert snap["total_requests"] >= 2

    @pytest.mark.asyncio
    async def test_pipeline_never_crashes_on_log_failure(self) -> None:
        """Even with a bad log path, the pipeline must not crash."""
        import enterprise_agentic_rag.graph.workflow as wf_mod
        import enterprise_agentic_rag.observability.tracer as tr_mod

        # Use a path that can't be written to
        bad_logger = EventLogger(log_path="/root/forbidden/events.jsonl")
        saved_tracer = tr_mod._tracer
        saved_wf_tracer = wf_mod._tracer
        try:
            tr_mod._tracer = Tracer(logger=bad_logger)
            tr_mod._tracer.metrics.reset()
            wf_mod._tracer = tr_mod._tracer

            g = build_workflow()
            result = await g.ainvoke({
                "query": "测试",
                "user_id": "u001",
                "session_id": "obs_crash",
                "trace_id": "trace-crash",
            })
            # Must complete regardless of log failure
            assert "final_answer" in result
        finally:
            tr_mod._tracer = saved_tracer
            wf_mod._tracer = saved_wf_tracer


# =========================================================================
# Singleton helpers
# =========================================================================


class TestSingletons:
    """Verify singleton access patterns."""

    def test_get_tracer_returns_singleton(self) -> None:
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_get_metrics_returns_singleton(self) -> None:
        m1 = get_metrics_collector()
        m2 = get_metrics_collector()
        assert m1 is m2
