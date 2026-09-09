"""Audio policy: no BGM per shot, one global BGM decision for the final cut.

Principles:
- Shot generation only allows diegetic audio (sound with a visible / physical source), no non-diegetic score.
- If BGM is wanted, it is mixed in as a single global music bed after the final concat.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

AUDIO_PROMPT_SUFFIX = (
    "Audio design: generate only diegetic, physically sourced sounds that belong to "
    "the visible scene or are explicitly implied by the shot, such as footsteps, rain, "
    "water, tools, room tone, audible on-screen dialogue, human speech, radio, ambient "
    "environmental sound, or music performed by clearly visible on-screen instruments or musicians. "
    "If a visible person is speaking, talking, whispering, shouting, being interviewed, narrating, "
    "or delivering lines, generate clear synchronized diegetic voice from that on-screen person; "
    "do not replace speech with silence, subtitles, room tone only, or vague ambience. Visible instruments "
    "should produce natural performance sound matching their action. Do not generate any non-diegetic "
    "background music, soundtrack, score, cinematic music bed, off-screen instrumental music, or unrelated "
    "BGM inside this clip."
)

NO_BGM_NEGATIVE_PROMPT = (
    "non-diegetic background music, unrelated BGM, soundtrack, musical score, cinematic score, "
    "music bed, off-screen music, invisible instrument music, music from unseen source, "
    "looped montage music, added background track"
)

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

_BGM_REQUIRED_TERMS = (
    "background music", "bgm", "music bed", "soundtrack", "musical score",
    "cinematic score", "non-diegetic music", "配乐", "背景音乐", "音乐床",
)

_BGM_OPT_OUT_TERMS = (
    "no music", "without music", "silent", "silence", "无配乐", "不要配乐", "无背景音乐",
)

_STYLE_KEYWORDS = (
    "vintage", "retro", "sentimental", "calm", "warm", "cinematic", "documentary",
    "tense", "suspense", "dramatic", "uplifting", "melancholic", "ambient",
    "soft", "minimal", "romantic", "epic", "electronic", "orchestral",
)

_SPEECH_TERMS = (
    "dialogue", "dialog", "speaking", "speak", "talking", "talk", "conversation",
    "interview", "narration", "narrator", "voice", "spoken", "speech", "monologue",
    "whisper", "shout", "says", "said", "speaks", "dialogue line", "delivering lines",
    "line delivery", "mouth movement", "lip movement", "说话", "对话", "讲话", "旁白", "采访",
    "人声", "台词", "独白", "开口",
)

_MUSIC_SOURCE_TERMS = (
    "instrument", "guitar", "piano", "violin", "cello", "drum", "saxophone", "trumpet",
    "flute", "harp", "keyboard", "synthesizer", "band", "orchestra", "musician",
    "playing music", "plays music", "singing", "singer", "song", "sing", "karaoke",
    "speaker", "speakers", "radio", "stereo", "boombox", "amplifier", "turntable",
    "乐器", "吉他", "钢琴", "小提琴", "鼓", "萨克斯", "乐队", "演奏", "音乐家",
    "唱歌", "歌手", "歌曲", "音响", "扬声器", "收音机", "留声机",
)

_VISIBLE_INSTRUMENT_TERMS = _MUSIC_SOURCE_TERMS


def _shot_text(shot: Dict[str, Any]) -> str:
    # Only the raw shot text feeds the audio semantics detection, so the generic constraints in
    # rewritten_prompt (e.g. "If any visible person speaks...") cannot pollute speech/music decisions.
    chunks = [
        shot.get("description_prompt", ""),
        str(shot.get("audio_label", "")),
    ]
    return "\n".join(str(c) for c in chunks if c).lower()


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if term.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text) is not None
    return term in text


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def audio_reference_type_for_shot(
    shot: Dict[str, Any],
    speech_reference_only: bool = True,
    music_reference_enabled: bool = True,
) -> Optional[str]:
    """Return the sound reference types available for this shot from the same label."""
    if shot.get("audio_label") is None:
        return None
    text = _shot_text(shot)
    has_speech = _contains_any(text, _SPEECH_TERMS)
    has_music_source = _contains_any(text, _MUSIC_SOURCE_TERMS)
    if has_speech:
        return "speech_voice"
    if music_reference_enabled and has_music_source:
        return "diegetic_music"
    if not speech_reference_only:
        return "general_diegetic"
    return None


def is_speech_reference_shot(shot: Dict[str, Any], speech_reference_only: bool = True) -> bool:
    """Decide whether this shot is suitable for extracting audio_url as a speaker-voice reference."""
    return audio_reference_type_for_shot(
        shot,
        speech_reference_only=speech_reference_only,
        music_reference_enabled=False,
    ) == "speech_voice"


def shot_has_explicit_music_source(shot: Dict[str, Any]) -> bool:
    """Decide whether the prompt explicitly features a diegetic music source."""
    return _contains_any(_shot_text(shot), _MUSIC_SOURCE_TERMS)


def apply_no_bgm_prompt(prompt: str, enabled: bool = True) -> str:
    """Add an explicit \"diegetic sound only, no BGM\" constraint to a shot generation prompt."""
    if not enabled:
        return prompt
    prompt = (prompt or "").strip()
    if not prompt:
        return AUDIO_PROMPT_SUFFIX
    if "non-diegetic background music" in prompt or "Do not generate any non-diegetic" in prompt:
        return prompt
    return f"{prompt}\n\n{AUDIO_PROMPT_SUFFIX}"


def build_no_bgm_negative_prompt(extra: Optional[str] = None, enabled: bool = True) -> Optional[str]:
    """Build a WAN negative_prompt that suppresses per-clip BGM."""
    if not enabled:
        return extra
    parts = [NO_BGM_NEGATIVE_PROMPT]
    if extra:
        parts.append(extra)
    return ", ".join(parts)


def _all_text(case_or_plan: Dict[str, Any]) -> str:
    chunks = [
        case_or_plan.get("overall_description_prompt", ""),
        case_or_plan.get("global_editing_style", ""),
        case_or_plan.get("title", ""),
    ]
    for shot in case_or_plan.get("shots", []) or []:
        chunks.append(shot.get("description_prompt", ""))
    return "\n".join(str(c) for c in chunks if c).lower()


def infer_bgm_style(case_or_plan: Dict[str, Any]) -> str:
    """Roughly extract a BGM style tag from the prompt / style text."""
    text = _all_text(case_or_plan)
    hits = [kw for kw in _STYLE_KEYWORDS if kw in text]
    if hits:
        return "_".join(dict.fromkeys(hits))
    return "general"


def prompt_requests_global_bgm(case_or_plan: Dict[str, Any]) -> bool:
    """Decide whether the prompt explicitly asks for non-diegetic BGM."""
    text = _all_text(case_or_plan)
    if any(term in text for term in _BGM_OPT_OUT_TERMS):
        return False
    return any(term in text for term in _BGM_REQUIRED_TERMS)


def _iter_audio_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths or []:
        root = Path(raw).expanduser()
        if root.is_file() and root.suffix.lower() in _AUDIO_EXTS:
            yield root
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in _AUDIO_EXTS:
                    yield path


def select_bgm_track(style: str, library_dirs: Iterable[str]) -> Optional[str]:
    """Pick a BGM from the library by style keyword; falls back to the first entry."""
    files = list(_iter_audio_files(library_dirs))
    if not files:
        return None
    style_terms = [s for s in (style or "").lower().replace("-", "_").split("_") if s]
    for path in files:
        name = path.stem.lower().replace("-", "_")
        if any(term in name for term in style_terms):
            return str(path)
    return str(files[0])


def decide_global_bgm(case_or_plan: Dict[str, Any], audio_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide whether the final cut gets a global BGM mixed in."""
    audio_cfg = audio_cfg or {}
    bgm_cfg = audio_cfg.get("global_bgm") or {}
    mode = str(bgm_cfg.get("mode", "auto")).lower()
    style = infer_bgm_style(case_or_plan)

    if mode in ("off", "none", "disable", "disabled", "false"):
        return {"enabled": False, "style": style, "reason": "global_bgm disabled by config"}

    explicit_source = bgm_cfg.get("source_path")
    if explicit_source:
        source = str(Path(explicit_source).expanduser())
        if Path(source).exists():
            return {"enabled": True, "style": style, "source_path": source, "reason": "explicit source_path"}
        return {"enabled": False, "style": style, "reason": f"configured source_path not found: {source}"}

    requested = prompt_requests_global_bgm(case_or_plan)
    if mode == "always":
        requested = True
    if not requested:
        return {"enabled": False, "style": style, "reason": "prompt does not explicitly request non-diegetic BGM"}

    source = select_bgm_track(style, bgm_cfg.get("library_dirs") or [])
    if not source:
        return {"enabled": False, "style": style, "reason": "BGM requested but no music asset found"}
    return {"enabled": True, "style": style, "source_path": source, "reason": "selected from library"}
