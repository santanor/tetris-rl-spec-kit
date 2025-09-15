"""Warning heuristics module stub.

Tests currently expect evaluate_warnings attribute to be absent; adding this
now will flip test to failing (driving next implementation stage).

Planned heuristics (to be implemented later):
- Low reward variance detection
- Hole count increasing trend
- Plateau of cleared lines
Will eventually expose evaluate_warnings(session: Session) -> list[Warning]
"""
from __future__ import annotations

from typing import List, Any

def evaluate_warnings(session: Any) -> List[str]:  # placeholder signature
    """Return an empty list for now.

    Later this will inspect session metrics and produce warning codes.
    """
    return []
