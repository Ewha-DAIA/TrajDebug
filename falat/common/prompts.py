"""Load the per-stage prompt files in ``falat/prompts/``.

Each YAML file holds one prompt:

    id                      key used by the stage code
    stage / source / notes  documentation
    verbatim                true if ``template`` is copied unchanged from the paper
    placeholders            names the template expects
    template                prompt text (paper text when verbatim: true)
    appended_output_format  optional, NOT from the paper; inserted right before the
                            template's final "Output only the JSON object." line
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import yaml

from .parsing import placeholders_in

_FINAL_LINE = "Output only the JSON object."


def _assemble(spec: Dict) -> str:
    template = spec["template"]
    extra = spec.get("appended_output_format")
    if not extra:
        return template
    body = template.rstrip("\n")
    if body.endswith(_FINAL_LINE):
        body = body[: -len(_FINAL_LINE)].rstrip("\n")
        return f"{body}\n{extra.rstrip()}\n{_FINAL_LINE}\n"
    return f"{body}\n{extra}"


def load_prompts(prompt_dir: Path) -> Dict[str, Dict]:
    """Return {id: {"text": assembled prompt, "file": ..., **spec}}."""
    prompts: Dict[str, Dict] = {}
    for path in sorted(Path(prompt_dir).glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        pid = spec["id"]
        if pid in prompts:
            raise ValueError(f"duplicate prompt id {pid!r} in {path.name}")
        missing = set(map(str, spec.get("placeholders", []))) ^ placeholders_in(spec["template"])
        if missing:
            raise ValueError(f"{path.name}: placeholders list does not match template: {sorted(missing)}")
        prompts[pid] = {**spec, "file": path.name, "text": _assemble(spec)}
    return prompts


def prompts_fingerprint(prompt_dir: Path) -> str:
    """Hash of all prompt files (recorded in config.json)."""
    h = hashlib.sha256()
    for path in sorted(Path(prompt_dir).glob("*.yaml")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
