"""Self-evaluation package for the editing agent: B1 transition timing / D2 transition effect / D3 transition audio-visual relation.

Fully isolated from benchmark: the evaluation logic is self-contained in transition_eval.py and
the expert-model microservices use offset ports (default +100, i.e. 8101/8104/8105/8107).
"""

from .transition_eval import (  # noqa: F401
    DEFAULT_THRESHOLDS,
    SERVICE_PORTS,
    ServiceUnavailable,
    build_plan_from_decisions,
    check_services,
    collect_failed_transitions,
    evaluate_video,
    inspect_clip_audio,
    inspect_transition_audio,
    require_services,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "SERVICE_PORTS",
    "ServiceUnavailable",
    "build_plan_from_decisions",
    "check_services",
    "collect_failed_transitions",
    "evaluate_video",
    "inspect_clip_audio",
    "inspect_transition_audio",
    "require_services",
]
