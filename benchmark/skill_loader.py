"""
Generic runtime loader for CutCraft VLM skill configuration files.

Skills live under benchmark/skills/<skill-name>/. Each skill directory has a
SKILL.md (human-readable documentation for the AI agent / developers) and,
optionally, a prompts.yaml file that is the SINGLE SOURCE OF TRUTH for that
skill's runtime prompts and tunable parameters.

Evaluation code should import this module and call load_skill(skill_name)
instead of hardcoding prompt strings, so that editing prompts.yaml actually
changes evaluation behavior without touching Python source.

Usage:
    from skill_loader import load_skill
    cfg = load_skill("e3-event-coherence-skill")
    system_prompt = cfg["labeling"]["system_prompt"]

Currently only e3-event-coherence-skill has a prompts.yaml wired into the
loader; the other 4 skills (shot-alignment-skill, montage-classification-skill,
event-fidelity-skill, transition-semantics-skill) are still documentation-only
and have no prompts.yaml yet.
"""
import os

import yaml

_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def get_skill_config_path(skill_name: str) -> str:
    """Return the expected prompts.yaml path for a given skill name."""
    return os.path.join(_SKILLS_DIR, skill_name, "prompts.yaml")


def load_skill(skill_name: str) -> dict:
    """Load and return the parsed prompts.yaml for a given skill.

    Args:
        skill_name: directory name under benchmark/skills/, e.g.
            "e3-event-coherence-skill".

    Returns:
        Parsed YAML content as a dict (empty dict if the file is empty).

    Raises:
        FileNotFoundError: if benchmark/skills/<skill_name>/prompts.yaml
            does not exist. Callers that need to tolerate missing configs
            (e.g. skills that have not been migrated to Plan B yet) should
            catch this and fall back to hardcoded defaults.
    """
    config_path = get_skill_config_path(skill_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Skill config not found: {config_path}\n"
            f"Expected benchmark/skills/{skill_name}/prompts.yaml to exist."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
