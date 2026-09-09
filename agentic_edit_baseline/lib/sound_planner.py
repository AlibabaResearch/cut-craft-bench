"""Shot-level sound planning and event-chain description.

Runs after Stage1 and before the Stage2 prompt rewrite:
- names the diegetic sound elements of each shot;
- describes how the previous shot's action sound continues into this one, and how this one leads into the next;
- emits a sound_design instruction that can be injected into the WAN prompt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")

_SOUND_PATTERNS = (
    ("chalk", "chalk dust / hand friction", "dry chalk rubbing, powder bursts, and soft chalk particles falling"),
    ("magnesium carbonate", "chalk dust / hand friction", "dry chalk rubbing, powder bursts, and soft chalk particles falling"),
    ("rub", "hand friction", "dry hand rubbing and granular skin-on-chalk friction"),
    ("footstep", "footsteps", "rubber-soled footsteps on the spring floor"),
    ("shoe", "shoe squeak", "soft rubber shoe squeaks and mat contact"),
    ("squeak", "shoe squeak", "brief rubber squeaks synchronized to foot movement"),
    ("jump", "jump takeoff", "short foot push-off and body lift"),
    ("catch", "ring catch", "hands clamping onto wooden rings with a tight leather creak"),
    ("clamp", "ring grip", "hands clamping onto wooden rings with a tight leather creak"),
    ("strap", "leather strap movement", "taut leather straps creaking, jerking, vibrating, and settling"),
    ("leather", "leather strap movement", "taut leather straps creaking, jerking, vibrating, and settling"),
    ("creak", "apparatus creak", "tense strap creaks under load"),
    ("metallic", "steel rig resonance", "faint metallic resonance from the overhead rig"),
    ("steel", "steel rig resonance", "faint metallic resonance from the overhead rig"),
    ("exhale", "breathing", "strained breathing and controlled exhale tied to physical effort"),
    ("breath", "breathing", "strained breathing and controlled exhale tied to physical effort"),
    ("landing", "landing impact", "deep damped thud from the foam mat and a soft chalk puff"),
    ("land", "landing impact", "deep damped thud from the foam mat and a soft chalk puff"),
    ("impact", "landing impact", "deep damped thud from the foam mat and a soft chalk puff"),
    ("mat", "foam mat contact", "muted foam compression, rebound, and fabric friction"),
    ("hum", "room tone", "broad gym room tone and faint overhead light hum"),
    ("ambient", "gym ambience", "large indoor gym ambience with soft reverberation"),
    ("ventilation", "ventilation ambience", "distant ventilation and cavernous gym air"),
    ("cough", "distant human ambience", "distant muffled coach cough and sparse hall activity"),
    ("dialogue", "on-screen speech", "clear diegetic human dialogue synchronized with the visible speaker's mouth movement"),
    ("speaking", "on-screen speech", "clear diegetic human speech synchronized with the visible speaker's mouth movement"),
    ("talking", "on-screen speech", "clear diegetic human speech synchronized with the visible speaker's mouth movement"),
    ("interview", "on-screen speech", "clear interview voice from the visible person in the scene"),
    ("narration", "on-screen speech", "clear diegetic narration or spoken voice tied to the visible speaker"),
    ("monologue", "on-screen speech", "clear diegetic monologue from the visible speaker"),
    ("whisper", "on-screen speech", "audible whispered human voice synchronized with the visible speaker"),
    ("说话", "on-screen speech", "清晰可听的画面内人声，与画面中人物口型和表演同步"),
    ("对话", "on-screen speech", "清晰可听的画面内对白，与画面中人物口型和表演同步"),
    ("台词", "on-screen speech", "清晰可听的画面内台词，与画面中人物口型和表演同步"),
    ("采访", "on-screen speech", "清晰可听的画面内采访人声，与说话人口型同步"),
    ("salute", "settling stillness", "breathing settling into a quiet gym room tone"),
)

_MUSIC_TERMS = (
    "music", "song", "singing", "instrument", "band", "speaker", "radio", "dj",
    "音乐", "歌曲", "唱歌", "乐器", "乐队", "音响", "收音机",
)

_SPEECH_TERMS = (
    "dialogue", "dialog", "speaking", "speak", "talking", "talk", "conversation",
    "interview", "narration", "narrator", "voice", "spoken", "speech", "monologue",
    "whisper", "shout", "says", "said", "speaks", "dialogue line", "delivering lines",
    "line delivery", "mouth movement", "lip movement", "说话", "对话", "讲话", "旁白",
    "采访", "人声", "台词", "独白", "开口",
)


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    low_term = term.lower()
    if low_term.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(low_term)}(?![a-z0-9])", text) is not None
    return low_term in text


def _contains_any(text: str, terms: List[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _contains_speech(text: str) -> bool:
    return _contains_any((text or "").lower(), list(_SPEECH_TERMS))


def _sound_pattern_matches(text: str, key: str) -> bool:
    """Keep non-Foley text such as catch/catches or \"landing each word\" from triggering gymnastics sounds."""
    if key == "landing" and re.search(r"\blanding\s+(?:each\s+)?(?:word|words|sentence|line|phrase)", text):
        return False
    return _contains_term(text, key)


def _strip_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:Shot|镜头)\s*\d+\s*\[[^\]]*\]\s*[:：]?\s*", "", text or "", flags=re.I).strip()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(_strip_prefix(text)) if s.strip()]


def _contains_music_source(text: str) -> bool:
    low = (text or "").lower()
    return _contains_any(low, list(_MUSIC_TERMS))


def _infer_elements(text: str) -> List[Dict[str, str]]:
    low = (text or "").lower()
    elements: List[Dict[str, str]] = []
    seen = set()
    for key, label, detail in _SOUND_PATTERNS:
        if _sound_pattern_matches(low, key) and label not in seen:
            elements.append({"name": label, "description": detail})
            seen.add(label)
    if _contains_speech(low) and "on-screen speech" not in seen:
        elements.append({
            "name": "on-screen speech",
            "description": "clear audible diegetic human speech synchronized with the visible speaker's mouth movement",
        })
        seen.add("on-screen speech")
    if "room tone" not in seen and "gym ambience" not in seen:
        elements.append({
            "name": "continuous room tone",
            "description": "consistent low-level indoor room tone matching the same physical space",
        })
    return elements


def _event_summary(shot: Dict[str, Any]) -> str:
    text = shot.get("description_prompt", "") or shot.get("rewritten_prompt", "") or ""
    for sentence in _sentences(text):
        low = sentence.lower()
        if any(k in low for k in ("rub", "jump", "catch", "swing", "hold", "release", "land", "salute", "impact")):
            return sentence
    sentences = _sentences(text)
    return sentences[0] if sentences else "this shot's visible physical action"


def _transition_audio_hint(transition: Optional[Dict[str, Any]]) -> str:
    if not transition:
        return "straight continuity of the same physical space"
    desc = transition.get("description") or ""
    relation = transition.get("audio_visual_relation") or transition.get("audio_relation") or ""
    timing = transition.get("timing_offset_seconds")
    parts = []
    if relation:
        parts.append(f"audio relation: {relation}")
    if timing not in (None, ""):
        parts.append(f"timing offset about {timing}s")
    if desc:
        parts.append(desc)
    return "; ".join(str(p) for p in parts if p) or "straight continuity of the same physical space"


def _format_elements(elements: List[Dict[str, str]]) -> str:
    return "; ".join(f"{e['name']}: {e['description']}" for e in elements)


def build_shot_sound_design(
    shot: Dict[str, Any],
    prev_shot: Optional[Dict[str, Any]],
    next_shot: Optional[Dict[str, Any]],
    overall_room_tone: str,
) -> Dict[str, Any]:
    text = str(shot.get("description_prompt", "") or "")
    elements = _infer_elements(text)
    event = _event_summary(shot)
    prev_event = _event_summary(prev_shot) if prev_shot else None
    next_event = _event_summary(next_shot) if next_shot else None
    from_previous = (
        f"This shot starts from the acoustic tail of the previous action ({prev_event}); preserve the same room tone and let any residual voice, contact, breath, object movement, or environmental sound decay naturally into this shot."
        if prev_shot else
        "This shot establishes the baseline acoustic space: consistent room tone, visible physical sounds, and no non-diegetic music."
    )
    to_next = (
        f"The ending sound should motivate the next visible action ({next_event}); carry the relevant physical or vocal tail as a subtle J/L-cut style diegetic continuation, not as added music."
        if next_shot else
        "The ending sound should settle into the same room tone and physical stillness without added music."
    )
    has_music_source = _contains_music_source(text)
    has_speech = _contains_speech(text)
    if has_music_source:
        music_guard = "If a visible physical music source is present, keep it diegetic and spatially consistent."
    elif has_speech:
        music_guard = (
            "No non-diegetic music should be generated; preserve clear diegetic speech/voice from the visible "
            "speaker, with natural room tone, Foley, breath, and environmental sound."
        )
    else:
        music_guard = "No music should be generated; use only room tone, Foley, breath, contact, apparatus, and environmental sound."
    instruction = (
        "Shot sound design: "
        f"core visible action is {event}. "
        f"Generate these diegetic sounds clearly: {_format_elements(elements)}. "
        f"Continuity from previous shot: {from_previous} "
        f"Continuity toward next shot: {to_next} "
        f"Maintain a consistent acoustic bed: {overall_room_tone}. {music_guard}"
    )
    return {
        "shot_id": shot.get("shot_id"),
        "event_summary": event,
        "diegetic_elements": elements,
        "room_tone": overall_room_tone,
        "continuity_from_previous": from_previous,
        "continuity_to_next": to_next,
        "transition_audio_hint": _transition_audio_hint(shot.get("transition_to_next")),
        "instruction": instruction,
    }


def enrich_sound_plan(plan: Dict[str, Any], enabled: bool = True) -> Dict[str, Any]:
    """Add sound_design and sound_event_chain to shot_plan in place."""
    if not enabled:
        return plan
    shots = plan.get("shots", []) or []
    overall_text = "\n".join(str(x) for x in (
        plan.get("overall_description_prompt"),
        plan.get("global_editing_style"),
    ) if x)
    overall_room_tone = "continuous low-level room tone matching the same physical location"
    if any(term in overall_text.lower() for term in ("gym", "gymnastics", "training hall", "体育馆", "训练馆")):
        overall_room_tone = "large gymnastics training hall room tone: distant ventilation, soft reverberation, faint overhead light hum, and subtle floor/mat resonance"
    elif any(term in overall_text.lower() for term in ("street", "outdoor", "街")):
        overall_room_tone = "continuous outdoor ambience matching the same street or exterior space"

    chain = []
    for idx, shot in enumerate(shots):
        prev_shot = shots[idx - 1] if idx > 0 else None
        next_shot = shots[idx + 1] if idx + 1 < len(shots) else None
        sound_design = build_shot_sound_design(shot, prev_shot, next_shot, overall_room_tone)
        shot["sound_description"] = sound_design["instruction"]
        shot["sound_design"] = sound_design
        chain.append({
            "shot_id": shot.get("shot_id"),
            "event_summary": sound_design["event_summary"],
            "continuity_to_next": sound_design["continuity_to_next"],
            "transition_audio_hint": sound_design["transition_audio_hint"],
        })
    plan["sound_event_chain"] = chain
    return plan
