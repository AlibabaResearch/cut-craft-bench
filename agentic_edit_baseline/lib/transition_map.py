"""Transition enum -> ffmpeg parameter mapping.

Normalises prompt.json's optical_effect / audio_visual_relation and maps them to an ffmpeg xfade
transition name plus an audio strategy, for use by Stage 5 (video_compose).

The enums in prompt.json (mixed case, so they need normalising):
  optical_effect: hard cut / dissolve / flash-to-white / flash-to-black / wipe
  audio_visual_relation: J-cut / L-cut / straight cut
  cinematographic_type: match on action / graphic match / wipe-by (occlusion) / ...
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---- Unified transition timing constants ----
# TRANSITION_EFFECT_SECONDS: one render duration for every optical transition (dissolve/flash black/flash white/wipe).
#   The GT transition_duration_seconds in prompt.json spread over 0~0.8s; an over-long dissolve/wipe
#   reads as a blurry passage to D2 and can eat into the net duration, while a too-short one is judged
#   a straight cut. 0.25s measured too short (TransNetV2 often calls it a hard cut), so 0.40s keeps the
#   effect out of the net duration while staying detectable. The GT value stays in the decision's gt_transition_duration_seconds.
TRANSITION_EFFECT_SECONDS = 0.40
# JL_CUT_OFFSET_SECONDS: one offset for J-cut / L-cut audio-visual displacement (the "extra time").
#   The GT spread over 0.2~0.9s and D3's signal arbitration calls a too-small offset straight; 0.8s puts
#   the speech / object sound projected across the cut well past D3_OBJECT_SYNC_TAIL_THRESHOLD (0.25s).
JL_CUT_OFFSET_SECONDS = 0.8


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


# optical_effect substring -> ffmpeg xfade transition name
# xfade transition=fade is a cross-dissolve; fadewhite/fadeblack are flash white/black
_OPTICAL_TO_XFADE = {
    "dissolve": "fade",
    "cross-dissolve": "fade",
    "flash-to-white": "fadewhite",
    "flash-to-black": "fadeblack",
    "fade": "fade",
    "wipe": "wipeleft",
}


def map_optical_effect(optical_effect: Optional[str]) -> Dict[str, Any]:
    """Return {mode, xfade}:
        mode = 'cut'   -> plain concat (hard cut)
        mode = 'xfade' -> use xfade, xfade holds the transition name
    """
    s = _norm(optical_effect)
    if not s or "hard cut" in s or "straight cut" in s:
        return {"mode": "cut", "xfade": None}
    for key, xfade in _OPTICAL_TO_XFADE.items():
        if key in s:
            return {"mode": "xfade", "xfade": xfade}
    # Unrecognised: fall back safely to a hard cut
    return {"mode": "cut", "xfade": None}


def map_audio_relation(audio_visual_relation: Optional[str]) -> str:
    """Normalise the audio-visual relation: 'j-cut' | 'l-cut' | 'straight'."""
    s = _norm(audio_visual_relation)
    if "j-cut" in s:
        return "j-cut"
    if "l-cut" in s:
        return "l-cut"
    return "straight"


def effect_duration_for(optical_effect: Optional[str]) -> float:
    """Effect duration actually rendered for this transition (seconds). Hard cut / unrecognised -> 0, otherwise 0.40s."""
    compose = map_optical_effect(optical_effect)
    if compose.get("mode") != "xfade":
        return 0.0
    return TRANSITION_EFFECT_SECONDS


def jl_offset_for(audio_visual_relation: Optional[str]) -> float:
    """J/L-cut audio-visual offset actually used (seconds). straight -> 0, J/L -> 0.8s."""
    if map_audio_relation(audio_visual_relation) in ("j-cut", "l-cut"):
        return JL_CUT_OFFSET_SECONDS
    return 0.0


def is_occlusion_transition(transition: Optional[Dict[str, Any]],
                            continuity_types: Optional[list] = None) -> bool:
    """Whether this transition needs last-frame-as-first-frame continuous generation.

    By default this matches wipe-by (foreground-occlusion) transitions.
    continuity_types: config.first_frame_continuity_types, matched case-insensitively as a substring.
    """
    if not transition:
        return False
    cine = _norm(transition.get("cinematographic_type"))
    keys = continuity_types if continuity_types else ["wipe-by", "foreground-occlusion", "遮挡"]
    return any(_norm(k) in cine for k in keys)


def overlap_seconds_for(transition_out: Optional[Dict[str, Any]]) -> float:
    """Seconds of visual overlap this transition needs (extra generation for the previous shot + stitching overlap).

    Only cross transitions that really overlap the picture (dissolve / wipe) count; hard cuts, flash
    white/black (stitched with a no-overlap concat) and wipe-by transitions (already joined at the first frame) return 0.

    The headroom equals the render duration TRANSITION_EFFECT_SECONDS (0.40s) exactly -- the previous
    shot generates that extra 0.40s and xfade eats it during stitching, so no net duration is lost.
    This covers the PICTURE overlap only; the audio headroom a J/L-cut needs is computed separately as
    generation_planner's audio_extension_seconds and is deliberately not mixed in (otherwise xfade would
    stretch to the offset time and render the effect as a 0.8s long dissolve).
    """
    if not transition_out:
        return 0.0
    if transition_out.get("occlusion_first_frame_handled"):
        return 0.0
    comp = transition_out.get("compose") or {}
    if comp.get("mode") != "xfade":
        return 0.0
    xf = comp.get("xfade") or ""
    if xf in ("fadewhite", "fadeblack"):  # flash white/black has no overlap, so no duration is consumed
        return 0.0
    return TRANSITION_EFFECT_SECONDS
