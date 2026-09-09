"""Stage 1.5 -- global consistency anchoring + narrative structure analysis.

Fixes environment / subject-id drift across shots. The main model wan2.7-t2v is text-only and
accepts no reference image (so identity cannot be anchored the way OpenMontage's reference_to_video
does), therefore this module runs one global LLM pass over the whole case before per-shot rewriting:

  - entities:   recurring subjects / characters, each with a STABLE and DETAILED appearance
                description (ID bible) reused verbatim by every related shot.
  - settings:   scenes / environments, each with a stable and detailed description (setting bible).
  - storylines: event-line partition. Edit instructions may be non-contiguous (cross-cutting /
                parallel narration), so shots of one event line can be interrupted by other lines.
  - shot_bindings: per-shot entity_ids / setting_ids / storyline_id + logical_prev_shot, i.e. the
                   PREVIOUS SHOT OF THE SAME EVENT LINE (which may skip physically adjacent shots
                   when cross-cutting), used as the narrative-continuity reference for that shot.

Fallback when the LLM is unavailable or parsing fails: a single event line, logical_prev = the
physically previous shot, no bible; the main flow is never blocked.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm_client import QwenClient

SYSTEM_PROMPT = (
    "You are a film script supervisor and continuity analyst for a multi-shot video "
    "sequence that will be generated shot-by-shot by a text-to-video model WITHOUT any "
    "reference-image conditioning. Your job is to guarantee cross-shot consistency of "
    "characters/subjects and environments, and to reveal the narrative structure "
    "(which may be non-linear, e.g. cross-cutting between parallel storylines).\n"
    "Analyze ALL shots together and output STRICT JSON only (no markdown, no prose) with keys:\n"
    "{\n"
    '  "entities": [ {"id": "snake_case_id", "name": "human label", '
    '"description": "ONE dense, stable, reusable appearance description: gender/age, '
    'body build, face shape, eye/nose/mouth details, hair style/color, skin tone, '
    'wardrobe with exact garment types, colors, patterns, fabrics, fit, footwear, '
    'distinctive props/accessories/textures. This exact description will be reused '
    'verbatim in every shot the entity appears, so make it self-contained, '
    'unambiguous, and especially specific about clothing."} ],\n'
    '  "settings": [ {"id": "snake_case_id", "name": "human label", '
    '"description": "ONE dense, stable description of the location: place type, key '
    'objects/layout, surface materials, lighting quality, color palette, time of day. '
    'Reused verbatim across shots sharing this setting."} ],\n'
    '  "storylines": [ {"id": "A", "description": "what this event line follows"} ],\n'
    '  "shot_bindings": [ {"shot_id": 1, "entity_ids": [...], "setting_ids": [...], '
    '"storyline_id": "A", "logical_prev_shot": null } ]\n'
    "}\n"
    "Rules:\n"
    "- Reuse the SAME entity/setting id across every shot where that subject/place recurs.\n"
    "- storyline_id groups shots of one continuous event line. If the sequence is linear, "
    "use a single storyline 'A'. If shots alternate between parallel actions (cross-cutting), "
    "assign different storyline ids (A/B/...).\n"
    "- logical_prev_shot = the shot_id of the PREVIOUS shot in the SAME storyline "
    "(NOT necessarily shot_id-1). The first shot of each storyline has null.\n"
    "- Output valid JSON parseable by json.loads. No trailing commas, no comments."
)


def _shot_digest(shots: List[Dict[str, Any]]) -> str:
    """Compress the key information of every shot into the analysis input."""
    lines = []
    for s in shots:
        sid = s.get("shot_id")
        desc = (s.get("description_prompt") or "").strip()
        trans = s.get("transition_to_next") or {}
        cine = trans.get("cinematographic_type", "")
        lines.append(f"Shot {sid}: {desc}"
                     + (f"\n  (transition to next: {cine})" if cine else ""))
    return "\n".join(lines)


def _build_user_prompt(plan: Dict[str, Any]) -> str:
    overall = plan.get("overall_description_prompt", "")
    style = plan.get("global_editing_style", "")
    digest = _shot_digest(plan.get("shots", []))
    return (
        f"[Overall sequence]\n{overall}\n\n"
        f"[Global editing style]\n{style}\n\n"
        f"[Shots ({len(plan.get('shots', []))})]\n{digest}\n\n"
        "[Task]\nProduce the continuity JSON described in the system message for THIS sequence."
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM output, tolerating ```json fences and surrounding noise."""
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    # fallback: take from the first { to the last }
    if not m:
        lo, hi = candidate.find("{"), candidate.rfind("}")
        if lo != -1 and hi != -1 and hi > lo:
            candidate = candidate[lo:hi + 1]
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _fallback_analysis(plan: Dict[str, Any]) -> Dict[str, Any]:
    """LLM unavailable: single event line, logical_prev = physically previous shot, no bible."""
    shots = plan.get("shots", [])
    bindings = []
    prev_sid = None
    for s in shots:
        sid = s.get("shot_id")
        bindings.append({
            "shot_id": sid,
            "entity_ids": [],
            "setting_ids": [],
            "storyline_id": "A",
            "logical_prev_shot": prev_sid,
        })
        prev_sid = sid
    return {
        "entities": [],
        "settings": [],
        "storylines": [{"id": "A", "description": "linear sequence (fallback)"}],
        "shot_bindings": bindings,
        "_fallback": True,
    }


def _normalize(analysis: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate / complete the LLM output: every shot has a binding and a legal logical_prev."""
    shots = plan.get("shots", [])
    valid_ids = [s.get("shot_id") for s in shots]

    entities = analysis.get("entities") or []
    settings = analysis.get("settings") or []
    storylines = analysis.get("storylines") or [{"id": "A", "description": ""}]

    raw_bindings = {b.get("shot_id"): b for b in (analysis.get("shot_bindings") or [])}

    # track the shots seen per storyline, used to fill in a missing logical_prev
    last_in_storyline: Dict[str, Any] = {}
    bindings: List[Dict[str, Any]] = []
    prev_sid = None
    for sid in valid_ids:
        b = raw_bindings.get(sid) or {}
        storyline = b.get("storyline_id") or "A"
        logical_prev = b.get("logical_prev_shot")
        # validate logical_prev; fall back to the previous shot of the same line, then the physically previous one
        if logical_prev not in valid_ids or logical_prev == sid:
            logical_prev = last_in_storyline.get(storyline, prev_sid)
        bindings.append({
            "shot_id": sid,
            "entity_ids": b.get("entity_ids") or [],
            "setting_ids": b.get("setting_ids") or [],
            "storyline_id": storyline,
            "logical_prev_shot": logical_prev,
        })
        last_in_storyline[storyline] = sid
        prev_sid = sid

    return {
        "entities": entities,
        "settings": settings,
        "storylines": storylines,
        "shot_bindings": bindings,
        "_fallback": analysis.get("_fallback", False),
    }


def analyze_case(
    plan: Dict[str, Any],
    client: Optional[QwenClient],
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Run consistency + narrative analysis over the whole case and return the normalised structure."""
    if client is None:
        return _fallback_analysis(plan)
    try:
        raw = client.chat(SYSTEM_PROMPT, _build_user_prompt(plan), max_tokens=max_tokens)
        parsed = _parse_json(raw)
        if not parsed or "shot_bindings" not in parsed:
            print("  [CONSISTENCY] LLM output cannot be parsed as the expected JSON, falling back to physical adjacency")
            return _fallback_analysis(plan)
        return _normalize(parsed, plan)
    except Exception as e:  # noqa: BLE001
        print(f"  [CONSISTENCY] analysis failed({e}), falling back to physical adjacency")
        return _fallback_analysis(plan)


# ---------------- lookup helpers used by prompt_rewriter ----------------

def build_lookup(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Build the id->description and shot_id->binding maps used during rewriting."""
    ent = {e.get("id"): e for e in analysis.get("entities", []) if e.get("id")}
    st = {s.get("id"): s for s in analysis.get("settings", []) if s.get("id")}
    bind = {b.get("shot_id"): b for b in analysis.get("shot_bindings", [])}
    return {"entities": ent, "settings": st, "bindings": bind}


def bible_for_shot(shot_id: Any, lookup: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return the detailed entity / setting descriptions bound to a shot (for prompt injection)."""
    b = lookup["bindings"].get(shot_id) or {}
    ent_desc = []
    for eid in b.get("entity_ids", []):
        e = lookup["entities"].get(eid)
        if e and e.get("description"):
            ent_desc.append(f"{e.get('name', eid)}: {e['description']}")
    set_desc = []
    for sid in b.get("setting_ids", []):
        s = lookup["settings"].get(sid)
        if s and s.get("description"):
            set_desc.append(f"{s.get('name', sid)}: {s['description']}")
    return {
        "entities": ent_desc,
        "settings": set_desc,
        "storyline_id": b.get("storyline_id", "A"),
        "logical_prev_shot": b.get("logical_prev_shot"),
    }
