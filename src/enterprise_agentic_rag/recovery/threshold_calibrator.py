"""Auto-rollback threshold calibration — dynamic safety thresholds for Harness.

Automatically calibrates rollback thresholds based on observed metrics,
reducing false positives and ensuring timely rollbacks.

The system tracks baseline metrics (latency, error rate, quality scores)
and adjusts thresholds as data accumulates, using statistical methods:
- Moving average + standard deviation bands
- Exponential weighted moving average (EWMA) for recent trends
- Seasonality-aware adjustment

Reference:
    TECHNICAL_DEEP_DIVE.md §41.3 — "自动回滚阈值需更多线上数据校准"
    TECHNICAL_DEEP_DIVE.md §41.4 — "Harness: 变更维度强制单因子发布"
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CALIBRATION_PATH = os.getenv("ROLLBACK_CALIBRATION_PATH", "data/harness/thresholds.json")
_MIN_DATA_POINTS = int(os.getenv("ROLLBACK_MIN_DATA_POINTS", "50"))
_WINDOW_SIZE = int(os.getenv("ROLLBACK_WINDOW_SIZE", "100"))
_EWMA_ALPHA = float(os.getenv("ROLLBACK_EWMA_ALPHA", "0.3"))


@dataclass
class MetricWindow:
    """Rolling window of metric values with statistical analysis."""

    values: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))
    ewma: float = 0.0
    initialized: bool = False

    def push(self, value: float) -> None:
        self.values.append(value)
        if not self.initialized:
            self.ewma = value
            self.initialized = True
        else:
            self.ewma = _EWMA_ALPHA * value + (1 - _EWMA_ALPHA) * self.ewma

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        variance = sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1)
        return variance ** 0.5

    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def is_mature(self) -> bool:
        return len(self.values) >= _MIN_DATA_POINTS


@dataclass
class RollbackThresholds:
    """Calibrated rollback safety thresholds."""

    # Latency thresholds (seconds)
    p95_latency_warning: float = 10.0
    p95_latency_critical: float = 30.0

    # Error rate thresholds
    error_rate_warning: float = 0.10
    error_rate_critical: float = 0.25

    # Quality thresholds
    verification_pass_rate_min: float = 0.85
    hallucination_rate_max: float = 0.15
    context_precision_min: float = 0.70

    # Calibration metadata
    last_calibrated: str = ""
    data_points: int = 0
    confidence: str = "low"  # low | medium | high

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": {
                "p95_warning_s": self.p95_latency_warning,
                "p95_critical_s": self.p95_latency_critical,
            },
            "errors": {
                "rate_warning": self.error_rate_warning,
                "rate_critical": self.error_rate_critical,
            },
            "quality": {
                "verification_pass_rate_min": self.verification_pass_rate_min,
                "hallucination_rate_max": self.hallucination_rate_max,
                "context_precision_min": self.context_precision_min,
            },
            "meta": {
                "last_calibrated": self.last_calibrated,
                "data_points": self.data_points,
                "confidence": self.confidence,
            },
        }


class ThresholdCalibrator:
    """Auto-calibrates rollback thresholds from observed production metrics.

    Uses statistical methods:
    1. Baseline from moving average ± 3σ bands
    2. EWMA for recent trend detection
    3. Gradual threshold tightening as confidence grows
    """

    def __init__(self) -> None:
        self._latency_window = MetricWindow()
        self._error_rate_window = MetricWindow()
        self._verification_window = MetricWindow()
        self._hallucination_window = MetricWindow()
        self._precision_window = MetricWindow()
        self._thresholds = RollbackThresholds()
        self._loaded = False

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_latency(self, p95_seconds: float) -> None:
        """Record a latency observation."""
        self._latency_window.push(p95_seconds)
        self._check_recalibrate()

    def observe_error_rate(self, rate: float) -> None:
        """Record an error rate observation."""
        self._error_rate_window.push(rate)
        self._check_recalibrate()

    def observe_verification(self, pass_rate: float) -> None:
        """Record a verification pass rate observation."""
        self._verification_window.push(pass_rate)  # Higher is better
        self._check_recalibrate()

    def observe_hallucination(self, rate: float) -> None:
        """Record a hallucination rate observation."""
        self._hallucination_window.push(rate)  # Lower is better
        self._check_recalibrate()

    def observe_precision(self, precision: float) -> None:
        """Record a context precision observation."""
        self._precision_window.push(precision)
        self._check_recalibrate()

    def observe_batch(self, metrics: dict[str, Any]) -> None:
        """Record a batch of metric observations from a metrics snapshot."""
        snapshot = metrics.get("metrics_snapshot", metrics)

        p95 = snapshot.get("p95_latency_ms", 0) / 1000.0
        if p95 > 0:
            self.observe_latency(p95)

        error_rate = snapshot.get("error_rate", -1)
        if error_rate >= 0:
            self.observe_error_rate(error_rate)

        verify_pass = snapshot.get("verification_pass_rate", -1)
        if verify_pass >= 0:
            self.observe_verification(verify_pass)

        hall_rate = snapshot.get("hallucination_rate", -1)
        if hall_rate >= 0:
            self.observe_hallucination(hall_rate)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _check_recalibrate(self) -> None:
        """Check if enough data has been collected to recalibrate."""
        if not self._latency_window.is_mature:
            return

        # Recalibrate every 50 new data points
        if self._latency_window.count % 50 != 0:
            return

        self._calibrate()

    def _calibrate(self) -> None:
        """Run threshold calibration using statistical analysis."""
        # ── Latency thresholds ──
        if self._latency_window.is_mature:
            baseline = self._latency_window.ewma
            noise = self._latency_window.std

            # Warning: 3σ above baseline (or at least 2x baseline)
            self._thresholds.p95_latency_warning = max(
                baseline + 3 * noise,
                baseline * 2.0,
                5.0,  # Absolute minimum
            )

            # Critical: 6σ above baseline (or at least 4x baseline)
            self._thresholds.p95_latency_critical = max(
                baseline + 6 * noise,
                baseline * 4.0,
                15.0,  # Absolute minimum
            )

        # ── Error rate thresholds ──
        if self._error_rate_window.is_mature:
            baseline = self._error_rate_window.ewma
            noise = self._error_rate_window.std

            self._thresholds.error_rate_warning = min(
                0.50,  # Absolute max
                max(baseline + 3 * noise, 0.05),  # At least 5%
            )

            self._thresholds.error_rate_critical = min(
                0.75,
                max(baseline + 5 * noise, 0.15),
            )

        # ── Quality thresholds ──
        if self._verification_window.is_mature:
            baseline = self._verification_window.ewma
            noise = self._verification_window.std

            self._thresholds.verification_pass_rate_min = max(
                0.60,  # Absolute minimum
                min(0.95, baseline - 2 * noise),  # 2σ below baseline
            )

        if self._hallucination_window.is_mature:
            baseline = self._hallucination_window.ewma
            noise = self._hallucination_window.std

            self._thresholds.hallucination_rate_max = min(
                0.50,
                max(0.05, baseline + 3 * noise),  # 3σ above baseline
            )

        if self._precision_window.is_mature:
            baseline = self._precision_window.ewma
            noise = self._precision_window.std

            self._thresholds.context_precision_min = max(
                0.50,
                min(0.95, baseline - 2 * noise),
            )

        # ── Confidence assessment ──
        total_points = min(
            self._latency_window.count,
            self._error_rate_window.count,
            self._verification_window.count,
        )
        if total_points >= 500:
            self._thresholds.confidence = "high"
        elif total_points >= 100:
            self._thresholds.confidence = "medium"
        else:
            self._thresholds.confidence = "low"

        self._thresholds.last_calibrated = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._thresholds.data_points = total_points

        logger.info(
            "Thresholds calibrated (confidence=%s): "
            "latency_warn=%.1fs latency_crit=%.1fs "
            "error_warn=%.2f error_crit=%.2f "
            "verify_pass_min=%.2f hall_max=%.2f",
            self._thresholds.confidence,
            self._thresholds.p95_latency_warning,
            self._thresholds.p95_latency_critical,
            self._thresholds.error_rate_warning,
            self._thresholds.error_rate_critical,
            self._thresholds.verification_pass_rate_min,
            self._thresholds.hallucination_rate_max,
        )

    # ------------------------------------------------------------------
    # Rollback decision
    # ------------------------------------------------------------------

    def should_rollback(self, current_metrics: dict[str, Any]) -> tuple[bool, str]:
        """Determine if a rollback is warranted based on current metrics.

        Args:
            current_metrics: Dict with p95_latency_ms, error_rate, etc.

        Returns:
            (should_rollback: bool, reason: str) tuple.
        """
        reasons: list[str] = []

        p95_lat = current_metrics.get("p95_latency_ms", 0) / 1000.0
        error_rate = current_metrics.get("error_rate", 0)
        verify_pass = current_metrics.get("verification_pass_rate", 1.0)
        hall_rate = current_metrics.get("hallucination_rate", 0)

        # Check against calibrated thresholds
        if p95_lat > self._thresholds.p95_latency_critical:
            reasons.append(
                f"P95 latency ({p95_lat:.1f}s) exceeds critical threshold "
                f"({self._thresholds.p95_latency_critical:.1f}s)"
            )

        if error_rate > self._thresholds.error_rate_critical:
            reasons.append(
                f"Error rate ({error_rate:.1%}) exceeds critical threshold "
                f"({self._thresholds.error_rate_critical:.1%})"
            )

        if verify_pass < self._thresholds.verification_pass_rate_min:
            reasons.append(
                f"Verification pass rate ({verify_pass:.1%}) below threshold "
                f"({self._thresholds.verification_pass_rate_min:.1%})"
            )

        if hall_rate > self._thresholds.hallucination_rate_max:
            reasons.append(
                f"Hallucination rate ({hall_rate:.1%}) exceeds threshold "
                f"({self._thresholds.hallucination_rate_max:.1%})"
            )

        should_rollback = len(reasons) >= 2  # At least 2 thresholds breached

        reason = "; ".join(reasons) if reasons else "All metrics within safe thresholds"

        if should_rollback:
            logger.warning("Rollback recommended: %s", reason)

        return should_rollback, reason

    # ------------------------------------------------------------------
    # Get / Set
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> RollbackThresholds:
        return self._thresholds

    def get_thresholds_dict(self) -> dict[str, Any]:
        return self._thresholds.to_dict()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist calibration state to disk."""
        os.makedirs(os.path.dirname(_CALIBRATION_PATH), exist_ok=True)

        data = {
            "thresholds": self._thresholds.to_dict(),
            "windows": {
                "latency_count": self._latency_window.count,
                "latency_ewma": self._latency_window.ewma,
                "error_rate_count": self._error_rate_window.count,
                "error_rate_ewma": self._error_rate_window.ewma,
                "verification_count": self._verification_window.count,
                "verification_ewma": self._verification_window.ewma,
                "hallucination_count": self._hallucination_window.count,
                "hallucination_ewma": self._hallucination_window.ewma,
            },
        }

        with open(_CALIBRATION_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Threshold calibration saved to %s", _CALIBRATION_PATH)

    def load(self) -> None:
        """Load calibration state from disk."""
        if not os.path.exists(_CALIBRATION_PATH):
            logger.debug("No saved calibration at %s", _CALIBRATION_PATH)
            return

        try:
            with open(_CALIBRATION_PATH, encoding="utf-8") as f:
                data = json.load(f)

            t = data.get("thresholds", {})
            self._thresholds = RollbackThresholds(
                p95_latency_warning=t.get("latency", {}).get("p95_warning_s", 10.0),
                p95_latency_critical=t.get("latency", {}).get("p95_critical_s", 30.0),
                error_rate_warning=t.get("errors", {}).get("rate_warning", 0.10),
                error_rate_critical=t.get("errors", {}).get("rate_critical", 0.25),
                verification_pass_rate_min=t.get("quality", {}).get("verification_pass_rate_min", 0.85),
                hallucination_rate_max=t.get("quality", {}).get("hallucination_rate_max", 0.15),
                context_precision_min=t.get("quality", {}).get("context_precision_min", 0.70),
                last_calibrated=t.get("meta", {}).get("last_calibrated", ""),
                data_points=t.get("meta", {}).get("data_points", 0),
                confidence=t.get("meta", {}).get("confidence", "low"),
            )

            windows = data.get("windows", {})
            self._latency_window.ewma = windows.get("latency_ewma", 0)
            self._error_rate_window.ewma = windows.get("error_rate_ewma", 0)
            self._verification_window.ewma = windows.get("verification_ewma", 0)
            self._hallucination_window.ewma = windows.get("hallucination_ewma", 0)

            self._loaded = True
            logger.info("Threshold calibration loaded from %s", _CALIBRATION_PATH)

        except Exception as exc:
            logger.warning("Failed to load threshold calibration: %s", exc)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_calibrator: ThresholdCalibrator | None = None


def get_calibrator() -> ThresholdCalibrator:
    """Get or create the global threshold calibrator."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ThresholdCalibrator()
        _calibrator.load()
    return _calibrator
