"""Stage 2 -- per-shot prompt rewriting.

Rewrites each shot's description_prompt + camera into a single-shot, transition-noise-free generation prompt:
  - preferably with Qwen3.7 (LLM agent), organised along OpenMontage's 5 layers
    (camera -> motion -> subject -> light -> style);
  - injects overall_description_prompt (global consistency) + global_editing_style (edit tone);
  - CONSISTENCY ANCHORING: injects the detailed bible descriptions of the entities / settings bound
    to this shot (from the consistency analysis), to be reused verbatim so ids and environments hold across shots;
  - NARRATIVE LINKAGE: the continuity anchor is the rewrite of the logical predecessor shot of the same
    event line (which may skip physically adjacent shots when cross-cutting), not simply the previous shot;
  - falls back to deterministic template assembly when the LLM is unavailable (camera + timestamp-stripped description + bible).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .llm_client import QwenClient
from .consistency import build_lookup, bible_for_shot

# Strip the leading "Shot N [xx-xx s]:" prefix from a description
_SHOT_PREFIX_RE = re.compile(r"^\s*(?:Shot|镜头)\s*\d+\s*\[[^\]]*\]\s*[:：]?\s*", re.IGNORECASE)


def _strip_shot_prefix(text: str) -> str:
    return _SHOT_PREFIX_RE.sub("", text or "").strip()


def _camera_phrase(camera: Dict[str, Any]) -> str:
    """Join the camera fields into one natural-language sentence (used by the fallback template and as an LLM hint)."""
    parts: List[str] = []
    for key in ("shot_scale", "angle", "camera_motion", "optical_motion", "depth_of_field"):
        val = camera.get(key)
        if val and str(val).lower() != "none":
            parts.append(str(val))
    return ", ".join(parts)


def _visual_only_anchor(text: str) -> str:
    """A same-label visual anchor keeps only appearance/wardrobe/environment, so the first shot's sound design does not propagate."""
    value = text or ""
    for marker in (". Diegetic sound design:", "\n\nShot-specific sound event chain:", "Audio design:"):
        idx = value.find(marker)
        if idx >= 0:
            value = value[:idx]
    sound_terms = (
        "sound", "audio", "ventilation", "cough", "squeak", "creak", "breath",
        "exhale", "thud", "hum", "noise", "foley", "room tone", "heard", "ring from",
    )
    sentences = re.split(r"(?<=[.!?])\s+", value.strip())
    visual_sentences = [
        sentence for sentence in sentences
        if not any(term in sentence.lower() for term in sound_terms)
    ]
    return " ".join(visual_sentences).strip() or value.strip()


SYSTEM_PROMPT = (
    "You are a professional cinematography prompt engineer for text-to-video models. "
    "Given ONE shot from a multi-shot sequence, rewrite it into a single self-contained "
    "generation prompt describing ONLY what happens WITHIN this shot. "
    "Follow a 5-layer structure and merge into fluent prose: "
    "(1) Camera (lens, depth of field), (2) Movement (shot size, camera movement), "
    "(3) Subject (action, appearance, texture), (4) Lighting (key, color temperature), "
    "(5) Style (mood adapted from the global editing style, not verbatim). "
    "CONSISTENCY (critical): The generator may have weak or no reference-image conditioning, "
    "so cross-shot identity/environment consistency depends heavily on your words. When entity "
    "or environment continuity descriptions are provided, you MUST weave in their EXACT appearance, "
    "face, hair, wardrobe, footwear, accessories, and setting details (same colors, build, materials, "
    "palette, lighting) so the subject and place look identical to the other shots. Clothing continuity "
    "is mandatory: repeat the same garment types, colors, patterns, fabrics, fit, and accessories unless "
    "the shot explicitly says the outfit changes. Do NOT invent conflicting attributes. "
    "When a same-label visual identity anchor is provided, use it ONLY to preserve the person's face, "
    "hair, body build, clothing, and accessories; do not copy its action, pose, camera, or background. "
    "When a previous shot on the same event line is provided, keep the subject and environment "
    "visually continuous with it. "
    "SOUND DESIGN (critical): If a shot-level sound design is provided, preserve it as concrete "
    "diegetic sound requirements. Explicitly describe visible/physically implied sounds, room tone, "
    "Foley, breath, object contact, and any acoustic tail from the previous event, while keeping all "
    "sound within this shot's physical scene. Do not turn these instructions into non-diegetic music. "
    "SPEECH (critical): If the shot description includes a visible person speaking, talking, whispering, "
    "shouting, being interviewed, narrating, or delivering dialogue/lines, explicitly require clear audible "
    "diegetic human speech synchronized with that person's mouth movement. Do not replace visible speech "
    "with silence, subtitles, room tone only, or vague ambience. "
    "SINGLE-SHOT CONTINUITY (critical): The output must describe exactly one uninterrupted camera take "
    "inside this generated clip. Do not introduce a second angle, inserted view, montage beat, internal "
    "cut, shot change, scene change, whip pan, snap zoom, rapid zoom, fast pan, fast tilt, slow dissolve, "
    "gradual dissolve, fade-through, morphing, cross-dissolve, or sudden camera move that could look like "
    "a transition. If the input camera movement is strong, rewrite it as smooth, slow, steady motion within "
    "one continuous shot. ADJACENT-SHOT DIFFERENCE (critical): preserve identity and setting continuity, but "
    "make this shot visibly different from adjacent shots through a distinct composition, action phase, camera "
    "distance, viewpoint, foreground/background arrangement, or subject position. When the adjacent shot's "
    "prompt is provided in the input, you MUST NOT replicate its composition, camera angle, framing, opening "
    "visual, or action phase — generate a CLEARLY DIFFERENT visual that any viewer would immediately recognize "
    "as a separate, distinct shot. Do NOT produce nearly identical or visually similar compositions between "
    "adjacent shots. If a same-label reference image is provided, use it only for identity and wardrobe; never "
    "copy its pose, framing, camera angle, background layout, or opening composition. "
    "STRICT RULES: Do NOT mention cuts, transitions, next/previous shots, timecodes, "
    "or editing terms (no 'L-cut', 'dissolve', 'match on action', etc.). "
    "Output ONLY the rewritten prompt text, no preamble, no quotes."
)


def _build_user_prompt(
    shot: Dict[str, Any],
    overall: str,
    global_style: str,
    bible: Optional[Dict[str, Any]],
    prev_line_prompt: Optional[str],
    continuity_hint: Optional[str],
    label_identity_anchor: Optional[str] = None,
    adjacent_prev_prompt: Optional[str] = None,
    adjacent_next_desc: Optional[str] = None,
) -> str:
    desc = _strip_shot_prefix(shot.get("description_prompt", ""))
    cam = _camera_phrase(shot.get("camera", {}))
    sound_design = shot.get("sound_design") or {}
    sound_instruction = shot.get("sound_description") or sound_design.get("instruction")
    lines = [
        f"[Overall sequence context]\n{overall}",
        f"[Global editing style / mood]\n{global_style}",
        f"[This shot description]\n{desc}",
        f"[Camera parameters]\n{cam}",
    ]
    if sound_instruction:
        lines.append(
            "[Shot-level sound design and event-chain audio continuity — preserve as diegetic audio requirements]\n"
            f"{sound_instruction}"
        )
    if bible and bible.get("entities"):
        lines.append(
            "[Consistent subjects in this shot — reuse these EXACT appearance details]\n"
            + "\n".join(f"- {d}" for d in bible["entities"])
        )
    if bible and bible.get("settings"):
        lines.append(
            "[Consistent environment in this shot — reuse these EXACT setting details]\n"
            + "\n".join(f"- {d}" for d in bible["settings"])
        )
    if label_identity_anchor:
        lines.append(
            "[Same-label visual identity anchor — preserve only face, hair, clothing, accessories]\n"
            f"{label_identity_anchor}"
        )
    if prev_line_prompt:
        lines.append(
            "[Previous shot on the SAME event line — stay visually continuous with it]\n"
            f"{prev_line_prompt}"
        )
    if adjacent_prev_prompt:
        lines.append(
            "[Physically ADJACENT previous shot — you MUST make this shot VISUALLY DIFFERENT from it]\n"
            f"{adjacent_prev_prompt}\n"
            "CRITICAL: This adjacent shot's composition, camera angle, framing, opening visual, "
            "action phase, and subject position are shown above. You MUST ensure this shot generates "
            "a CLEARLY DIFFERENT visual: use a different camera distance (e.g., close-up vs wide), "
            "different angle (e.g., high vs low), different framing, different action moment, or "
            "different subject screen position. Do NOT produce a nearly identical or visually similar "
            "composition. The viewer must immediately perceive these as two distinct shots."
        )
    if adjacent_next_desc:
        lines.append(
            "[Physically ADJACENT next shot — you MUST make this shot VISUALLY DIFFERENT from it]\n"
            f"{adjacent_next_desc}\n"
            "CRITICAL: The next shot's description is shown above. Ensure this shot's composition, "
            "camera angle, and visual setup are clearly distinct from what the next shot will show."
        )
    if continuity_hint:
        lines.append(
            "[Continuity anchor]\n"
            f"This shot should visually/aurally continue from: {continuity_hint} "
            "(reflect the continuation in the opening moment, but describe it as part of "
            "this shot, not as a transition)."
        )
    lines.append(
        "[Task]\nRewrite into a single self-contained generation prompt for THIS shot only, "
        "keeping subjects and environment consistent with the details above. Explicitly include "
        "stable face, hair, clothing colors, garment types, materials, footwear, and accessories "
        "for recurring same-label people. Also include a clear diegetic sound design sentence for "
        "this shot: visible physical sounds, continuous room tone, Foley/action sync, and the acoustic "
        "tail or lead-in implied by the event chain. If any visible person speaks, talks, whispers, shouts, "
        "is interviewed, narrates, or delivers dialogue/lines, explicitly require audible synchronized "
        "diegetic human speech from that person; never replace visible speech with silence, subtitles, "
        "room tone only, or vague ambience. Enforce single-shot continuity: the prompt must read "
        "as one uninterrupted camera take only, with no internal shot change, no inserted second angle, "
        "no montage, no slow dissolve, no gradual dissolve, no fade-through, no morphing, no sudden fast "
        "camera movement, and no camera motion that resembles a transition. Also enforce adjacent-shot "
        "difference: this shot must be visibly distinct from its neighbors in composition, action phase, "
        "camera distance, viewpoint, or subject position while preserving the same identity and setting. "
        "When an adjacent shot's prompt is provided, explicitly avoid replicating its composition, framing, "
        "opening visual, camera angle, or action phase — the two shots must look unmistakably different to any viewer."
    )
    return "\n\n".join(lines)


def _fallback_rewrite(
    shot: Dict[str, Any],
    global_style: str,
    bible: Optional[Dict[str, Any]] = None,
    label_identity_anchor: Optional[str] = None,
) -> str:
    """Deterministic template assembly: stripped description + camera phrase + bible consistency text + a short style hint."""
    desc = _strip_shot_prefix(shot.get("description_prompt", ""))
    cam = _camera_phrase(shot.get("camera", {}))
    parts = [p for p in (cam, desc) if p]
    text = ". ".join(parts)
    # Append the consistency bible (so cross-shot id / environment detail survives without an LLM)
    if bible:
        for d in bible.get("entities", []):
            text += f". {d}"
        for d in bible.get("settings", []):
            text += f". {d}"
    if label_identity_anchor:
        text += (
            ". Same-label identity anchor for this recurring subject: preserve the same face, hair, "
            f"body build, clothing, footwear, and accessories as: {label_identity_anchor}"
        )
    sound_design = shot.get("sound_design") or {}
    sound_instruction = shot.get("sound_description") or sound_design.get("instruction")
    if sound_instruction:
        element_names = [str(e.get("name")) for e in sound_design.get("diegetic_elements", []) if e.get("name")]
        if element_names:
            text += ". Diegetic sound design: " + ", ".join(dict.fromkeys(element_names)) + "; maintain consistent room tone and event-chain acoustic continuity"
        else:
            text += ". Diegetic sound design: maintain clear physical Foley, consistent room tone, and event-chain acoustic continuity"
    text += ". If any visible person speaks, talks, whispers, shouts, is interviewed, narrates, or delivers dialogue, generate clear audible diegetic human speech synchronized with that person's mouth movement; do not replace visible speech with silence, subtitles, room tone only, or vague ambience"
    text += ". Single-shot continuity: one uninterrupted camera take only, no internal cut, no inserted angle, no montage, no slow dissolve, no gradual dissolve, no fade-through, no morphing, no sudden fast pan, no whip pan, no snap zoom, and no rapid camera movement that resembles a transition. Adjacent-shot difference: keep the same identity and setting but use a clearly different composition, action phase, camera distance, viewpoint, or subject position from neighboring shots; do not copy a reference image's framing or opening composition"
    if global_style:
        mood = global_style.split(".")[0].strip()
        if mood:
            text = f"{text}. Style: {mood}"
    return text.strip()


def rewrite_shot(
    shot: Dict[str, Any],
    overall: str,
    global_style: str,
    client: Optional[QwenClient],
    bible: Optional[Dict[str, Any]] = None,
    prev_line_prompt: Optional[str] = None,
    continuity_hint: Optional[str] = None,
    label_identity_anchor: Optional[str] = None,
    adjacent_prev_prompt: Optional[str] = None,
    adjacent_next_desc: Optional[str] = None,
) -> str:
    """Rewrite a single shot. Falls back to the template when client is None or the call fails."""
    if client is not None:
        try:
            user_prompt = _build_user_prompt(
                shot, overall, global_style, bible, prev_line_prompt, continuity_hint,
                label_identity_anchor=label_identity_anchor,
                adjacent_prev_prompt=adjacent_prev_prompt,
                adjacent_next_desc=adjacent_next_desc,
            )
            out = client.chat(SYSTEM_PROMPT, user_prompt)
            if out and len(out) > 10:
                return out
            print(f"  [REWRITE] shot {shot.get('shot_id')} LLM output too short, falling back to the template")
        except Exception as e:  # noqa: BLE001
            print(f"  [REWRITE] shot {shot.get('shot_id')} LLM failed({e}), falling back to the template")
    return _fallback_rewrite(
        shot, global_style, bible,
        label_identity_anchor=label_identity_anchor,
    )


def rewrite_plan(
    plan: Dict[str, Any],
    client: Optional[QwenClient],
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rewrite the rewritten_prompt field of every shot in plan, in place.

    analysis: output of consistency.analyze_case. When provided, consistency comes from the entity/setting
              bible + the logical predecessor of the same event line; otherwise the anchor degrades to the previous shot's transition description.
    """
    overall = plan.get("overall_description_prompt", "")
    global_style = plan.get("global_editing_style", "")
    shots = plan.get("shots", [])

    lookup = build_lookup(analysis) if analysis else None
    shots_by_id: Dict[Any, Dict[str, Any]] = {s.get("shot_id"): s for s in shots}

    prev_transition_desc: Optional[str] = None
    label_identity_anchors: Dict[str, str] = {}
    prev_physical_rewritten: Optional[str] = None  # rewrite of the physically previous shot, used to demand an explicit difference
    for idx, shot in enumerate(shots):
        sid = shot.get("shot_id")
        bible = bible_for_shot(sid, lookup) if lookup else None

        # Narrative continuity reference: prefer the rewrite of the logical predecessor on the same event line
        prev_line_prompt = None
        if bible and bible.get("logical_prev_shot") is not None:
            prev_shot = shots_by_id.get(bible["logical_prev_shot"])
            if prev_shot:
                prev_line_prompt = prev_shot.get("rewritten_prompt")

        # Description of the physically adjacent shot, used to demand a visibly different frame
        adjacent_prev_prompt = prev_physical_rewritten
        adjacent_next_desc = None
        if idx + 1 < len(shots):
            next_shot = shots[idx + 1]
            adjacent_next_desc = _strip_shot_prefix(
                next_shot.get("description_prompt", ""))

        label_key = str(shot.get("audio_label")) if shot.get("audio_label") is not None else None
        label_identity_anchor = label_identity_anchors.get(label_key) if label_key else None

        rewritten = rewrite_shot(
            shot, overall, global_style, client,
            bible=bible,
            prev_line_prompt=prev_line_prompt,
            continuity_hint=prev_transition_desc,
            label_identity_anchor=label_identity_anchor,
            adjacent_prev_prompt=adjacent_prev_prompt,
            adjacent_next_desc=adjacent_next_desc,
        )
        shot["rewritten_prompt"] = rewritten
        if label_key and label_key not in label_identity_anchors:
            label_identity_anchors[label_key] = _visual_only_anchor(rewritten)
            shot["label_identity_anchor_role"] = "source"
        elif label_key:
            shot["label_identity_anchor_role"] = "follower"
        # Record consistency metadata to make the artifacts auditable
        if bible:
            shot["storyline_id"] = bible.get("storyline_id")
            shot["logical_prev_shot"] = bible.get("logical_prev_shot")

        # Prepare the "physically previous shot" continuity anchor (fallback path)
        trans = shot.get("transition_to_next") or {}
        prev_transition_desc = trans.get("description")
        # Record this shot's rewrite so the next shot can be asked to differ from it
        prev_physical_rewritten = rewritten
    return plan
