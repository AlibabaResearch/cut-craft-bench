"""Normalisation of the path fields in config.yaml.

No machine-specific absolute path is hardcoded in the config file:
  - in-repo paths are written relative to agentic_edit_baseline/ (e.g. ../prompt/prompt.json);
  - out-of-repo paths (model weights, output data disks) are written as ${ENV_NAME} or
    ${ENV_NAME:-default} placeholders: export them before running, or override with an absolute path.

load_config() reads the yaml and then calls resolve_config_paths(cfg), which expands the
placeholders and ~ and turns relative paths absolute, so downstream stages always get usable paths.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base directory for relative paths in config.yaml, i.e. agentic_edit_baseline/
BASE_DIR = Path(__file__).resolve().parent.parent

# ${NAME} and ${NAME:-default}. os.path.expandvars does not know the :- default syntax, so expand it here.
_VAR_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand_vars(value: str) -> Optional[str]:
    """Expand ${NAME} / ${NAME:-default}.

    A ${NAME} without a default returns None when unset, rather than splicing in an empty string --
    otherwise "${EDIT_BASELINE_OUT}/wan/artifacts" would become a plausible-looking
    "/wan/artifacts" and the error would only surface, cryptically, much later.
    """
    missing = False

    def sub(match: "re.Match[str]") -> str:
        nonlocal missing
        name, default = match.group(1), match.group(2)
        env = os.environ.get(name)
        if env:
            return env
        if default is None:
            missing = True
            return ""
        return default

    expanded = _VAR_RE.sub(sub, value)
    return None if missing else expanded


def resolve_path(value: Optional[str]) -> Optional[str]:
    """Expand ${ENV} / ${ENV:-default} / ~, then make relative paths absolute against BASE_DIR.

    Returns None when a required, unexported environment variable is referenced; the caller decides how to report it.
    """
    if not value:
        return value
    expanded = _expand_vars(str(value))
    if not expanded:
        return None if expanded is None else expanded
    expanded = os.path.expanduser(expanded)
    path = Path(expanded)
    if not path.is_absolute():
        path = BASE_DIR / path
    return os.path.normpath(str(path))


def resolve_config_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise every path field in cfg in place and return the same cfg for chaining."""
    if not isinstance(cfg, dict):
        return cfg

    paths = cfg.get("paths")
    if isinstance(paths, dict):
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = resolve_path(value)

    bgm = (cfg.get("audio") or {}).get("global_bgm") if isinstance(cfg.get("audio"), dict) else None
    if isinstance(bgm, dict):
        if isinstance(bgm.get("source_path"), str):
            bgm["source_path"] = resolve_path(bgm["source_path"])
        dirs: List[Any] = bgm.get("library_dirs") or []
        if isinstance(dirs, list):
            bgm["library_dirs"] = [
                resolve_path(d) if isinstance(d, str) else d for d in dirs
            ]

    return cfg
