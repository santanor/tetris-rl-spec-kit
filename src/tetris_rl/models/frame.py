from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Frame:
    step: int
    action: int
    reward: float
    board_metrics: Dict[str, Any]
    action_label: Optional[str] = None
    rgb_ref: Optional[str] = None  # path or identifier for stored frame
