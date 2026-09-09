"""Stage 3 -- generation planning.

The previous shot's transition type decides this shot's generation mode, producing edit_decisions:
  - wipe-by (foreground-occlusion) transition: this shot uses i2v with the previous shot's last frame
    as first frame, so the picture is seamless while the occluder sweeps past;
  - every other transition: this shot is generated independently with t2v and the join is left to Stage 5 (ffmpeg).

Per-shot structure of edit_decisions:
  {
    shot_id, duration, prompt, gen_mode ('t2v'|'i2v'),
    first_frame_from_shot,   # = the previous shot_id for i2v, otherwise None
    transition_out {         # the transition after this shot (None for the last shot)
        cinematographic_type, optical_effect, audio_visual_relation,
        timing_offset_seconds, transition_duration_seconds,
        compose {mode, xfade}, audio_relation
    }
  }
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .transition_map import (
    JL_CUT_OFFSET_SECONDS,
    TRANSITION_EFFECT_SECONDS,
    effect_duration_for,
    is_occlusion_transition,
    jl_offset_for,
    map_audio_relation,
    map_optical_effect,
    overlap_seconds_for,
)
from .audio_policy import (
    apply_no_bgm_prompt,
    audio_reference_type_for_shot,
    build_no_bgm_negative_prompt,
    is_speech_reference_shot,
    shot_has_explicit_music_source,
)
from .scene_audio_beds import (
    beds_for_shot,
    build_scene_audio_beds,
    build_scene_bed_prompt,
)


def _build_sound_design_prompt(sound_design: Optional[Dict[str, Any]]) -> str:
    if not sound_design:
        return ""
    instruction = sound_design.get("instruction") or ""
    elements = sound_design.get("diegetic_elements") or []
    element_names = [str(e.get("name")) for e in elements if e.get("name")]
    parts = [
        "Shot-specific sound event chain:",
        instruction,
    ]
    if element_names:
        parts.append("Required diegetic sound elements: " + ", ".join(dict.fromkeys(element_names)) + ".")
    if sound_design.get("continuity_from_previous"):
        parts.append("Audio continuity from previous action: " + str(sound_design.get("continuity_from_previous")))
    if sound_design.get("continuity_to_next"):
        parts.append("Audio continuity toward following action: " + str(sound_design.get("continuity_to_next")))
    return "\n".join(p for p in parts if p)


DEFAULT_SINGLE_SHOT_PROMPT = (
    "Single-shot visual continuity requirement: generate this clip as exactly one continuous camera take. "
    "Keep one stable viewpoint and one continuous camera movement within the shot. Do not create any internal "
    "cut, shot change, scene change, transition, montage beat, jump cut, cross-dissolve, slow dissolve, "
    "gradual dissolve, fade-through, morphing transition, whip pan, snap zoom, rapid zoom, fast pan, fast tilt, "
    "or sudden camera move that could look like a transition to another shot. Adjacent-shot differentiation (critical): "
    "this shot must have a clearly distinct composition, action phase, camera distance, or viewing angle from "
    "the previous and following shots. Do NOT produce a nearly identical or visually similar composition to any "
    "adjacent shot — the viewer must immediately perceive each shot as unmistakably different. Use different "
    "camera distance (close-up vs wide), different angle (high vs low), different framing, different action moment, "
    "or different subject screen position. If a reference image is used, use it only for subject identity and wardrobe; "
    "do not copy the reference image's pose, framing, camera angle, background layout, or opening composition."
)

DEFAULT_SINGLE_SHOT_NEGATIVE = (
    "internal cut, in-shot cut, shot change within clip, scene change within clip, extra shot, extra camera angle, "
    "montage, jump cut, fast cut, hard cut, cross dissolve inside clip, slow dissolve inside clip, gradual dissolve, "
    "fade transition inside clip, morphing transition, hidden transition, transition inside clip, whip pan, snap zoom, "
    "rapid zoom, fast pan, fast tilt, sudden camera movement, aggressive camera movement, copied reference framing, "
    "same opening composition as reference image, identical adjacent shot composition, nearly identical adjacent shot, "
    "visually similar neighboring shot, same camera angle as adjacent shot, same framing as previous shot, "
    "duplicate composition, repeated visual setup"
)


def _transition_offset(transition_out: Optional[Dict[str, Any]]) -> float:
    if not transition_out:
        return 0.0
    try:
        return max(0.0, float(transition_out.get("timing_offset_seconds") or 0))
    except (TypeError, ValueError):
        return 0.0


def _incoming_j_offset(prev_transition: Optional[Dict[str, Any]]) -> float:
    """J-cut offset entering this shot (how many seconds early this shot's audio must start).

    prev_transition may be the plan's raw shot.transition_to_next (carrying the GT offset) or an
    already normalised transition_out; both are renormalised to 0.8s through jl_offset_for so the
    raw GT value never disagrees with the actual Stage5 offset.
    """
    if not prev_transition:
        return 0.0
    relation = prev_transition.get("audio_visual_relation") or prev_transition.get("audio_relation")
    return jl_offset_for(relation) if map_audio_relation(relation) == "j-cut" else 0.0


def _outgoing_l_offset(transition_out: Optional[Dict[str, Any]]) -> float:
    if not transition_out or transition_out.get("audio_relation") != "l-cut":
        return 0.0
    return _transition_offset(transition_out)


def _build_single_shot_guard(guard_cfg: Optional[Dict[str, Any]]) -> Dict[str, str]:
    cfg = guard_cfg or {}
    if not cfg.get("enabled", True):
        return {"prompt": "", "negative": ""}
    return {
        "prompt": str(cfg.get("prompt") or DEFAULT_SINGLE_SHOT_PROMPT),
        "negative": str(cfg.get("negative_prompt") or DEFAULT_SINGLE_SHOT_NEGATIVE),
    }


def _append_extra(base: Optional[str], extra: Optional[str]) -> Optional[str]:
    if not extra:
        return base
    return f"{base}, {extra}" if base else extra


def plan_generation(
    plan: Dict[str, Any],
    first_frame_continuity_types: Optional[List[str]] = None,
    wan_min_duration: int = 2,
    wan_max_duration: int = 15,
    suppress_clip_bgm: bool = True,
    negative_prompt_extra: Optional[str] = None,
    speech_reference_only: bool = True,
    audio_cfg: Optional[Dict[str, Any]] = None,
    single_shot_guard: Optional[Dict[str, Any]] = None,
    audio_headroom_seconds: float = 1.0,
) -> Dict[str, Any]:
    """Build edit_decisions from shot_plan.

    audio_headroom_seconds: extra audio generated / stored per shot (seconds, 1.0 by default).
      It only lengthens the gen_duration sent to the model and the full track kept by Stage5, not
      net_duration, so the runtime is unchanged. The point is that a later D3 repair widening the
      J/L-cut offset has audio to shift, instead of regenerating a whole shot for a few hundred ms.
    """
    shots = plan.get("shots", [])
    scene_audio_beds = build_scene_audio_beds(plan, audio_cfg or {})
    shot_guard = _build_single_shot_guard(single_shot_guard)
    decisions: List[Dict[str, Any]] = []

    first_audio_source_by_label: Dict[str, Any] = {}
    first_visual_source_by_label: Dict[str, Any] = {}

    for idx, shot in enumerate(shots):
        prev_shot = shots[idx - 1] if idx > 0 else None
        # the previous shot's transition_to_next decides how this shot is entered
        prev_transition = prev_shot.get("transition_to_next") if prev_shot else None

        gen_mode = "t2v"
        first_frame_from = None
        if prev_transition and is_occlusion_transition(prev_transition, first_frame_continuity_types):
            gen_mode = "i2v"
            first_frame_from = prev_shot.get("shot_id")

        # the transition after this shot (used by Stage5 stitching)
        trans = shot.get("transition_to_next")
        transition_out = None
        if trans:
            # Transition timing normalisation: the GT transition_duration_seconds / timing_offset_seconds
            # are spread widely (0~0.9s); an over-long dissolve/wipe is judged a blurry transition by D2 and
            # a too-small J/L offset is judged straight by the D3 signal arbitration. Both are rewritten to
            # a 0.40s effect duration + 0.8s audio-visual offset; the GT values stay in gt_* for auditing.
            gt_effect_duration = trans.get("transition_duration_seconds") or 0
            gt_timing_offset = trans.get("timing_offset_seconds") or 0
            effect_duration = effect_duration_for(trans.get("optical_effect"))
            timing_offset = jl_offset_for(trans.get("audio_visual_relation"))
            transition_out = {
                "cinematographic_type": trans.get("cinematographic_type"),
                "optical_effect": trans.get("optical_effect"),
                "audio_visual_relation": trans.get("audio_visual_relation"),
                "timing_offset_seconds": timing_offset,
                "transition_duration_seconds": effect_duration,
                "gt_timing_offset_seconds": gt_timing_offset,
                "gt_transition_duration_seconds": gt_effect_duration,
                "effect_duration_policy": f"unified_{TRANSITION_EFFECT_SECONDS:g}s",
                "jl_offset_policy": f"unified_{JL_CUT_OFFSET_SECONDS:g}s",
                "compose": map_optical_effect(trans.get("optical_effect")),
                "audio_relation": map_audio_relation(trans.get("audio_visual_relation")),
                # wipe-by: the join already happens on the generation side, so the stitching duration is 0 (hard join)
                "occlusion_first_frame_handled": is_occlusion_transition(
                    trans, first_frame_continuity_types
                ),
            }
            # overlapping transitions (dissolve/wipe) need extra material: this shot generates overlap seconds more, eaten by the overlap
            transition_out["overlap_seconds"] = overlap_seconds_for(transition_out)

        # generation duration = net duration + visual overlap + J/L audio offset headroom (rounded up for WAN).
        base_duration = shot.get("duration")
        overlap_out = transition_out["overlap_seconds"] if transition_out else 0
        incoming_j = _incoming_j_offset(prev_transition)
        outgoing_l = _outgoing_l_offset(transition_out)
        outgoing_offset = _transition_offset(transition_out)
        # The audio headroom must cover both ends: a J-cut shifts this shot's audio incoming_j seconds
        # earlier (so the tail needs incoming_j extra seconds to reach the end of the picture), and an
        # L-cut drags it outgoing_l seconds later. Both can happen at once, hence the sum, not the max.
        audio_extension = max(incoming_j + outgoing_l, outgoing_offset)
        # An extra 0.5s guard band leaves room for J/L-cut offsetting, fades and the final trim.
        audio_extension_with_guard = audio_extension + 0.5 if audio_extension > 0 else 0.0
        # Required duration (headroom excluded): the lower bound of clip length the stitching really needs.
        required_clip_seconds = float(base_duration or 0) + max(
            float(overlap_out or 0), audio_extension_with_guard)
        # One more audio headroom slice (1s by default) is only generated and stored, never part of the
        # required duration, so a D3 repair can retune the audio offset on the existing clip.
        headroom = max(0.0, float(audio_headroom_seconds or 0))
        needed_duration = required_clip_seconds + headroom
        # gen_duration is the integer-second duration actually sent to the model and must satisfy the
        # provider's min/max duration limits (a 3s per-task floor, say). It is fully decoupled from
        # base_duration/net_duration (the edited net length) -- Stage5 trims the clip back to net_duration,
        # so a stretched gen_duration never makes the final cut run long.
        gen_duration = min(int(math.ceil(needed_duration)), int(wan_max_duration))
        gen_duration = max(gen_duration, int(wan_min_duration))

        shot_scene_beds = beds_for_shot(scene_audio_beds, shot.get("shot_id"))

        # Audio reference planning: voices and diegetic music sources can both reference the same label, but ambience with no visible music source is not reused.
        audio_label = shot.get("audio_label")
        audio_reference = None
        reference_type = audio_reference_type_for_shot(
            shot,
            speech_reference_only=speech_reference_only,
            music_reference_enabled=True,
        )
        if shot_scene_beds and not reference_type:
            reference_type = "scene_audio_bed"
        if audio_label is not None and reference_type:
            label_key = f"{audio_label}:{reference_type}"
            if label_key not in first_audio_source_by_label:
                first_audio_source_by_label[label_key] = shot.get("shot_id")
                audio_reference = {
                    "label": audio_label,
                    "role": "source",
                    "source_shot_id": shot.get("shot_id"),
                    "use_as_reference": False,
                    "reference_type": reference_type,
                }
            else:
                audio_reference = {
                    "label": audio_label,
                    "role": "follower",
                    "source_shot_id": first_audio_source_by_label[label_key],
                    "use_as_reference": True,
                    "reference_type": reference_type,
                }

        # Visual reference planning: the first shot of a label is the identity/frame source, later shots reference its first frame.
        visual_label = shot.get("audio_label")
        visual_reference = None
        if visual_label is not None:
            visual_key = str(visual_label)
            if visual_key not in first_visual_source_by_label:
                first_visual_source_by_label[visual_key] = shot.get("shot_id")
                visual_reference = {
                    "label": visual_label,
                    "role": "source",
                    "source_shot_id": shot.get("shot_id"),
                    "use_as_reference": False,
                    "reference_type": "same_label_first_frame",
                }
            else:
                visual_reference = {
                    "label": visual_label,
                    "role": "follower",
                    "source_shot_id": first_visual_source_by_label[visual_key],
                    "use_as_reference": True,
                    "reference_type": "same_label_first_frame",
                }

        prompt = shot.get("rewritten_prompt") or shot.get("description_prompt", "")
        sound_design = shot.get("sound_design") or {}
        sound_prompt = _build_sound_design_prompt(sound_design)
        if sound_prompt:
            prompt = f"{prompt}\n\n{sound_prompt}"
        prompt = apply_no_bgm_prompt(prompt, enabled=suppress_clip_bgm)
        scene_bed_prompt = build_scene_bed_prompt(shot_scene_beds)
        if scene_bed_prompt:
            prompt = f"{prompt}\n\n{scene_bed_prompt}"
        if shot_guard.get("prompt"):
            prompt = f"{prompt}\n\n{shot_guard['prompt']}"
        no_explicit_music = not shot_has_explicit_music_source(shot)
        has_scene_diegetic_music = any(bed.get("type") == "diegetic_music" for bed in shot_scene_beds)
        has_speech = is_speech_reference_shot(shot, speech_reference_only=speech_reference_only)
        if has_speech:
            prompt = (
                f"{prompt}\n\n"
                "Speech requirement for this shot: if a visible person is speaking, talking, whispering, shouting, "
                "being interviewed, narrating, or delivering dialogue/lines in this shot, generate clear audible "
                "diegetic human speech synchronized with their visible mouth movement and performance. The voice must "
                "belong to the on-screen speaker in the physical scene. Do not replace the dialogue with silence, room "
                "tone only, music, subtitles, or vague ambient sound. Keep the speech natural and intelligible while "
                "still avoiding non-diegetic BGM."
            )
        negative_extra = _append_extra(negative_prompt_extra, shot_guard.get("negative"))
        if suppress_clip_bgm and no_explicit_music and not has_scene_diegetic_music:
            if has_speech:
                prompt = (
                    f"{prompt}\n\n"
                    "Audio restriction for this speaking shot: the prompt contains no visible music-producing source, "
                    "so do not add non-diegetic BGM, soundtrack, score, or unrelated music. However, visible on-screen "
                    "speech is required: preserve clear diegetic dialogue/voice from the speaking person, plus natural "
                    "room tone, Foley, breath, and physical ambience."
                )
                speech_no_bgm = (
                    "non-diegetic background music, cinematic score, soundtrack, looped BGM, off-screen music, "
                    "instrumental backing track unrelated to the visible speaker"
                )
                negative_extra = _append_extra(negative_extra, speech_no_bgm)
            else:
                prompt = (
                    f"{prompt}\n\n"
                    "Audio restriction for this shot: the prompt contains no visible instrument, singer, "
                    "radio, speaker, or other music-producing source. Generate no music of any kind; "
                    "use only non-musical diegetic ambience, Foley, room tone, and physical sound effects."
                )
                hard_no_music = "non-diegetic background music, cinematic score, soundtrack, looped BGM, off-screen instrumental music, instrumental backing track, unrelated melody"
                negative_extra = _append_extra(negative_extra, hard_no_music)
        elif suppress_clip_bgm and has_scene_diegetic_music:
            scene_guard = (
                "Do not add unrelated soundtrack, cinematic score, or non-diegetic BGM; "
                "only preserve the specified scene-source diegetic music."
            )
            negative_extra = _append_extra(negative_extra, scene_guard)
        negative_prompt = build_no_bgm_negative_prompt(
            negative_extra, enabled=suppress_clip_bgm,
        )

        decisions.append({
            "shot_id": shot.get("shot_id"),
            "duration": base_duration,          # net duration (backward compatible)
            "net_duration": base_duration,      # effective duration of this shot after stitching
            "gen_duration": gen_duration,       # duration actually sent to WAN (includes overlap / audio headroom)
            "audio_extension_seconds": audio_extension_with_guard,
            "audio_headroom_seconds": headroom,           # extra audio headroom generated
            "required_clip_seconds": round(required_clip_seconds, 4),  # lower bound of clip length required by stitching
            "incoming_j_offset_seconds": incoming_j,
            "outgoing_l_offset_seconds": outgoing_l,
            "overlap_out_seconds": overlap_out, # visual overlap of the transition after this shot
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "gen_mode": gen_mode,
            "first_frame_from_shot": first_frame_from,
            "visual_reference": visual_reference,
            "audio_reference": audio_reference,
            "sound_design": sound_design,
            "scene_audio_beds": shot_scene_beds,
            "transition_out": transition_out,
        })

    return {
        "id": plan.get("id"),
        "title": plan.get("title", ""),
        "number_of_shots": plan.get("number_of_shots"),
        "total_duration": plan.get("total_duration"),
        "scene_audio_beds": scene_audio_beds,
        "decisions": decisions,
    }
