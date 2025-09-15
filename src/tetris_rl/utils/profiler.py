from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class _Section:
    total: float = 0.0
    calls: int = 0

    def add(self, dt: float):
        self.total += dt
        self.calls += 1

    @property
    def avg(self) -> float:
        return self.total / self.calls if self.calls else 0.0

@dataclass
class Profiler:
    """Lightweight interval profiler.

    Tracks cumulative and interval timings per named section. Intended for very low
    overhead coarse profiling inside the training loop.
    """
    interval_steps: int = 0
    sections: Dict[str, _Section] = field(default_factory=dict)
    interval_sections: Dict[str, _Section] = field(default_factory=dict)
    last_report_step: int = 0

    def record(self, name: str, dt: float):
        if name not in self.sections:
            self.sections[name] = _Section()
        if name not in self.interval_sections:
            self.interval_sections[name] = _Section()
        self.sections[name].add(dt)
        self.interval_sections[name].add(dt)

    def maybe_report(self, global_step: int) -> Dict[str, Any] | None:
        if self.interval_steps <= 0:
            return None
        if (global_step - self.last_report_step) < self.interval_steps:
            return None
        self.last_report_step = global_step
        total_interval = sum(s.total for s in self.interval_sections.values()) or 1.0
        report = {
            "type": "profile",
            "global_step": global_step,
            "interval_steps": self.interval_steps,
            "interval": {
                name: {
                    "total": round(sec.total, 6),
                    "calls": sec.calls,
                    "avg": round(sec.avg, 6),
                    "pct": round(100.0 * sec.total / total_interval, 2),
                } for name, sec in sorted(self.interval_sections.items())
            },
            "cumulative": {
                name: {
                    "total": round(sec.total, 6),
                    "calls": sec.calls,
                    "avg": round(sec.avg, 6),
                } for name, sec in sorted(self.sections.items())
            }
        }
        # reset interval
        self.interval_sections = {}
        return report

class SectionTimer:
    """Context manager to time a code block and record to profiler."""
    def __init__(self, profiler: Profiler | None, name: str):
        self.profiler = profiler
        self.name = name
        self.t0 = 0.0
    def __enter__(self):
        if self.profiler:
            self.t0 = time.perf_counter()
    def __exit__(self, exc_type, exc, tb):
        if self.profiler:
            self.profiler.record(self.name, time.perf_counter() - self.t0)

__all__ = ["Profiler", "SectionTimer"]
