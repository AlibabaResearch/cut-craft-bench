"""Scene-level diegetic sound-source planning.

Identifies diegetic sound that must persist across shots (dance-battle music, club PA, street
bands, a radio) and emits scene_audio_beds for generation planning and the final mix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .audio_policy import select_bgm_track

_DIEGETIC_MUSIC_TERMS = (
    "dance battle", "battle dance", "dance-off", "dance off", "dancing to music",
    "dance floor", "club", "nightclub", "dj", "speaker", "speakers", "sound system",
    "boombox", "radio", "stereo", "band", "concert", "street performance",
    "busker", "musician", "playing music", "dancing to a beat", "music beat", "hip-hop dance",
    "斗舞", "舞蹈比赛", "尬舞", "跳舞", "舞池", "夜店", "迪厅", "dj", "音响",
    "扬声器", "收音机", "留声机", "乐队", "演奏", "街头表演", "节拍", "节奏",
)

_SOURCE_HINT_TERMS = (
    "speaker", "speakers", "sound system", "boombox", "radio", "stereo", "dj",
    "band", "musician", "instrument", "音响", "扬声器", "收音机", "乐队", "乐器",
)

_DEFAULT_DIEGETIC_MUSIC_PROMPT = (
    "Scene audio continuity: the same diegetic music is continuously playing from the "
    "scene's physical sound source throughout this shot. It is not an added soundtrack "
    "or non-diegetic BGM. Keep the dancers' or characters' motion synchronized to the "
    "same beat and maintain a consistent venue sound across adjacent shots."
)


def _text_chunks(plan: Dict[str, Any]) -> List[str]:
    chunks = [
        plan.get("title", ""),
        plan.get("overall_description_prompt", ""),
        plan.get("global_editing_style", ""),
    ]
    for shot in plan.get("shots", []) or []:
        chunks.extend([
            shot.get("description_prompt", ""),
            str(shot.get("audio_label", "")),
        ])
    return [str(c) for c in chunks if c]


def _shot_text(shot: Dict[str, Any]) -> str:
    chunks = [
        shot.get("description_prompt", ""),
        str(shot.get("audio_label", "")),
    ]
    return "\n".join(str(c) for c in chunks if c).lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms if term)


def _shot_id(shot: Dict[str, Any], fallback: int) -> Any:
    return shot.get("shot_id", fallback)


def build_scene_audio_beds(
    plan: Dict[str, Any],
    audio_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Plan cross-shot diegetic sound sources from the semantics of the whole case."""
    audio_cfg = audio_cfg or {}
    if audio_cfg.get("allow_scene_diegetic_music", True) is False:
        return []
    bed_cfg = audio_cfg.get("scene_audio_beds") or {}
    if not bed_cfg.get("enabled", True):
        return []

    shots = plan.get("shots", []) or []
    if len(shots) < 2:
        return []

    all_text = "\n".join(_text_chunks(plan)).lower()
    if not bed_cfg.get("auto_detect", True) and not bed_cfg.get("source_path"):
        return []
    if not _contains_any(all_text, _DIEGETIC_MUSIC_TERMS) and not bed_cfg.get("source_path"):
        return []

    source_semantics = "visible or implied physical scene sound source"
    if _contains_any(all_text, ("dance", "dancing", "斗舞", "跳舞", "舞池", "dancing to a beat", "music beat")):
        source_semantics = "dance battle / visible or implied venue speaker"
    elif _contains_any(all_text, ("band", "concert", "乐队", "演奏", "街头表演")):
        source_semantics = "live performance / visible musicians"
    elif _contains_any(all_text, ("radio", "boombox", "speaker", "收音机", "音响", "扬声器")):
        source_semantics = "visible or implied playback device"

    explicit_shot_ids = [
        _shot_id(shot, idx + 1)
        for idx, shot in enumerate(shots)
        if _contains_any(_shot_text(shot), _DIEGETIC_MUSIC_TERMS + _SOURCE_HINT_TERMS)
    ]
    if len(explicit_shot_ids) >= 2:
        applies_to = explicit_shot_ids
    else:
        applies_to = [_shot_id(shot, idx + 1) for idx, shot in enumerate(shots)]

    source_path = bed_cfg.get("source_path")
    if source_path:
        source_path = str(Path(source_path).expanduser())
    elif bed_cfg.get("library_dirs"):
        source_path = select_bgm_track("dance_electronic_upbeat", bed_cfg.get("library_dirs") or [])

    mix_rule = {
        "volume": float(bed_cfg.get("volume", 0.35)),
        "fade_in_seconds": float(bed_cfg.get("fade_in_seconds", 0.2)),
        "fade_out_seconds": float(bed_cfg.get("fade_out_seconds", 0.4)),
        "target_lufs": float(bed_cfg.get("target_lufs", -18)),
    }

    bed: Dict[str, Any] = {
        "id": "diegetic_music_bed_1",
        "type": "diegetic_music",
        "source_semantics": source_semantics,
        "applies_to_shots": applies_to,
        "continuity": "continuous",
        "prompt_rule": bed_cfg.get("prompt_rule") or _DEFAULT_DIEGETIC_MUSIC_PROMPT,
        "mix_rule": mix_rule,
        "source_path": source_path,
        "reason": "detected scene-level diegetic music semantics",
    }
    if not source_path:
        bed["mix_status"] = "planned_only_no_source_path"
    return [bed]


def beds_for_shot(scene_audio_beds: Iterable[Dict[str, Any]], shot_id: Any) -> List[Dict[str, Any]]:
    """Return the scene_audio_beds covering the given shot."""
    result = []
    shot_key = str(shot_id)
    for bed in scene_audio_beds or []:
        applies = {str(x) for x in bed.get("applies_to_shots", [])}
        if shot_key in applies:
            result.append(bed)
    return result


def build_scene_bed_prompt(scene_beds: Iterable[Dict[str, Any]]) -> str:
    """Build the scene sound-continuity note injected into a shot prompt."""
    rules = [bed.get("prompt_rule") for bed in scene_beds or [] if bed.get("prompt_rule")]
    return "\n\n".join(dict.fromkeys(rules))
