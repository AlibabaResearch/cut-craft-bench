"""D3 pre-check audio gate.

D3 (transition audio-visual relation) can only hold when a clip really carries loud enough sound:
  - a J-cut needs recognisable diegetic sound at the START of the incoming shot;
  - an L-cut needs recognisable diegetic sound at the END of the outgoing shot;
  - a straight cut still needs a sound event near the cut as sync evidence.

Generation models (the r2v path especially) often return digital silence (-90 dBFS) or
near-silence (-47 dBFS); no audio bias or prompt tweak can then make D3 anything but straight.
So a gate runs BEFORE D3 scoring:
  1. measure each clip's overall / head / tail loudness locally (ffmpeg + librosa, no expert service);
  2. decide which shots lack loud enough sound;
  3. append a stronger-sound requirement to their prompts and regenerate (capped by max_regen_attempts).

This module only detects, decides and edits prompts; agent_loop performs the regeneration and restitch.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .transition_map import map_audio_relation

AUDIO_GATE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "min_full_dbfs": -40.0,
    "min_boundary_dbfs": -45.0,
    "boundary_window_seconds": 1.0,
    "max_regen_attempts": 1,
}

# Reinforcement prompts for the three missing-sound roles. The wording is deliberately specific:
# only diegetic sound is requested, no non-diegetic BGM, to avoid clashing with suppress_clip_bgm.
_SOUND_BOOST_WHOLE = (
    "MANDATORY AUDIO REQUIREMENT — this shot must not be silent: generate clearly audible, "
    "continuous diegetic sound for the entire shot, produced by what is visible in frame "
    "(speech from the on-screen person, footsteps, impacts, friction, breathing, equipment, "
    "vehicle or machine sound, water, crowd). The sound must be loud and present throughout, "
    "at normal foreground recording level, never near-silence and never an empty track. "
    "Do not add non-diegetic background music or score."
)
_SOUND_BOOST_HEAD = (
    "MANDATORY AUDIO TIMING — the very first moment of this shot must already be loud: "
    "a clearly audible diegetic sound owned by this shot (on-screen speech onset, footstep, "
    "impact, or action sound) must start at 0.0 seconds and stay continuously audible for at "
    "least the first 2 seconds, with no silent lead-in, so it can be heard before the cut."
)
_SOUND_BOOST_TAIL = (
    "MANDATORY AUDIO TIMING — the final moments of this shot must stay loud: a clearly audible "
    "diegetic sound owned by this shot (on-screen speech, sustained action, ringing/decaying "
    "object sound) must remain continuously audible through the last 2 seconds and still be "
    "sounding at the very end of the shot, with no fade to silence, so it can carry across the cut."
)


def _gate_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(AUDIO_GATE_DEFAULTS)
    merged.update(cfg or {})
    return merged


def shot_sound_roles(decisions: List[Dict[str, Any]]) -> Dict[Any, Dict[str, bool]]:
    """Derive which end of each shot needs sound, from the J/L-cut relations.

    shot i needs sound at its tail  <- its outgoing transition is an l-cut (sound continues into the next shot)
    shot i needs sound at its head  <- the previous transition is a j-cut (this shot's sound is heard early)
    """
    roles: Dict[Any, Dict[str, bool]] = {}
    for idx, dec in enumerate(decisions):
        trans = dec.get("transition_out") or {}
        prev_trans = decisions[idx - 1].get("transition_out") if idx > 0 else None
        out_rel = map_audio_relation(trans.get("audio_visual_relation")
                                    or trans.get("audio_relation")) if trans else "straight"
        in_rel = map_audio_relation((prev_trans or {}).get("audio_visual_relation")
                                    or (prev_trans or {}).get("audio_relation")) if prev_trans else "straight"
        roles[dec["shot_id"]] = {
            "needs_tail_sound": out_rel == "l-cut",
            "needs_head_sound": in_rel == "j-cut",
        }
    return roles


def inspect_shots(decisions: List[Dict[str, Any]],
                  clip_path_for: Callable[[Any], str],
                  inspect_fn: Callable[..., Dict[str, Any]],
                  cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Measure the loudness of every shot clip and return a gate report.

    inspect_fn: injected measurement function (agent_eval.transition_eval.inspect_clip_audio);
                this module does not depend on the evaluation package, so tests can swap it out.
    """
    gate = _gate_cfg(cfg)
    roles = shot_sound_roles(decisions)
    window = float(gate["boundary_window_seconds"])
    min_full = float(gate["min_full_dbfs"])
    min_boundary = float(gate["min_boundary_dbfs"])

    shots: List[Dict[str, Any]] = []
    for dec in decisions:
        shot_id = dec["shot_id"]
        role = roles.get(shot_id, {})
        report = inspect_fn(clip_path_for(shot_id),
                            head_window=window, tail_window=window,
                            silence_dbfs=min_boundary)
        full_dbfs = report.get("full_dbfs")
        head_dbfs = report.get("head_dbfs")
        tail_dbfs = report.get("tail_dbfs")

        reasons: List[str] = []
        if not report.get("has_audio", False):
            reasons.append("clip has no audio track or the audio is empty")
        elif full_dbfs is None:
            reasons.append("cannot read the loudness")
        else:
            if float(full_dbfs) < min_full:
                reasons.append(f"overall loudness {float(full_dbfs):.1f} dBFS < {min_full:.1f}, effectively silent")
            if role.get("needs_head_sound") and head_dbfs is not None \
                    and float(head_dbfs) < min_boundary:
                reasons.append(f"J-cut needs sound at the head, but the first {window:g}s is "
                               f"{float(head_dbfs):.1f} dBFS")
            if role.get("needs_tail_sound") and tail_dbfs is not None \
                    and float(tail_dbfs) < min_boundary:
                reasons.append(f"L-cut needs sound at the tail, but the last {window:g}s is "
                               f"{float(tail_dbfs):.1f} dBFS")

        shots.append({
            "shot_id": shot_id,
            "full_dbfs": full_dbfs,
            "head_dbfs": head_dbfs,
            "tail_dbfs": tail_dbfs,
            "first_voiced_time": report.get("first_voiced_time"),
            "last_voiced_time": report.get("last_voiced_time"),
            "needs_head_sound": bool(role.get("needs_head_sound")),
            "needs_tail_sound": bool(role.get("needs_tail_sound")),
            "silent": bool(reasons),
            "reasons": reasons,
        })

    failing = [s for s in shots if s["silent"]]
    return {
        "enabled": bool(gate["enabled"]),
        "thresholds": {"min_full_dbfs": min_full, "min_boundary_dbfs": min_boundary,
                       "boundary_window_seconds": window},
        "shots": shots,
        "silent_shot_ids": [s["shot_id"] for s in failing],
        "n_silent": len(failing),
        "n_total": len(shots),
    }


def apply_sound_boost(decisions: List[Dict[str, Any]], gate_report: Dict[str, Any],
                      attempt: int = 1) -> List[Dict[str, Any]]:
    """Append a stronger-sound requirement to silent shots' prompts and return the shots to regenerate.

    Repeated triggers on the same shot do not append the same text twice (that would inflate the prompt),
    but the shot is still listed for regeneration (another sampling pass may come back with sound).
    """
    by_id = {d["shot_id"]: d for d in decisions}
    boosted: List[Dict[str, Any]] = []
    for shot in gate_report.get("shots", []):
        if not shot.get("silent"):
            continue
        dec = by_id.get(shot["shot_id"])
        if dec is None:
            continue

        additions: List[str] = []
        # whole segment silent -> reinforce globally; only one end too quiet -> reinforce that end
        whole_silent = any("effectively silent" in r or "no audio track" in r or "cannot read" in r
                           for r in shot.get("reasons", []))
        if whole_silent:
            additions.append(_SOUND_BOOST_WHOLE)
        if shot.get("needs_head_sound") and any("at the head" in r for r in shot.get("reasons", [])):
            additions.append(_SOUND_BOOST_HEAD)
        if shot.get("needs_tail_sound") and any("at the tail" in r for r in shot.get("reasons", [])):
            additions.append(_SOUND_BOOST_TAIL)
        if not additions:
            additions.append(_SOUND_BOOST_WHOLE)

        existing = dec.setdefault("audio_gate_additions", [])
        new_parts = [a for a in additions if a not in existing]
        if new_parts:
            dec["prompt"] = dec["prompt"] + "\n\n" + "\n\n".join(new_parts)
            existing.extend(new_parts)
        boosted.append({
            "shot_id": shot["shot_id"],
            "attempt": attempt,
            "reasons": shot.get("reasons", []),
            "prompt_boosted": bool(new_parts),
            "full_dbfs": shot.get("full_dbfs"),
        })
    return boosted


def summarize(gate_report: Dict[str, Any]) -> str:
    parts = []
    for s in gate_report.get("shots", []):
        flag = "SILENT" if s["silent"] else "ok"
        parts.append(f"shot{s['shot_id']}={s.get('full_dbfs')}dBFS[{flag}]")
    return " ".join(parts)
