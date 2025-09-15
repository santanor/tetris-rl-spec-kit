"""Observation mode wrappers registry stub.

Empty by design for red->green progression. Tests import the module and expect
get_observation_mode to be missing currently.

Planned features:
- get_observation_mode(mode_name: str)
- BaseObservationAdapter protocol
- Implementations: 'features', 'grid', 'rgb_eval'
"""

from __future__ import annotations

from typing import Callable, Any, Dict

# Internal registry mapping mode names to builder callables (to be populated later)
_registry: Dict[str, Callable[..., Any]] = {}

def register_observation_mode(name: str, builder: Callable[..., Any]) -> None:
    if name in _registry:
        raise ValueError(f"Observation mode '{name}' already registered")
    _registry[name] = builder

def get_observation_mode(name: str) -> Any:
    """Return a callable/build object for the given observation mode.

    Current phase: raises NotImplementedError if unknown.
    """
    try:
        return _registry[name]
    except KeyError:
        raise NotImplementedError(f"Observation mode '{name}' is not implemented yet") from None

# Register a minimal dummy mode for testing
register_observation_mode("dummy", lambda x: x)
