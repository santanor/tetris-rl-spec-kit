from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, List
from collections import deque

@dataclass
class MetricStream:
    window: int = 50
    values: List[float] = field(default_factory=list)
    _dq: Deque[float] = field(default_factory=deque)
    _sum: float = 0.0

    def append(self, v: float) -> None:
        self.values.append(v)
        self._dq.append(v)
        self._sum += v
        if len(self._dq) > self.window:
            self._sum -= self._dq.popleft()

    def rolling_mean(self) -> float:
        if not self._dq:
            return 0.0
        return self._sum / len(self._dq)
