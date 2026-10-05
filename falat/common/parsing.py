"""Parsers for FALAT stage outputs and prompt rendering."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

TRANSITION_LABELS = ("follow_up", "redundancy", "no_influence", "error_shift", "correction", "dead_end")

# Matches {question}, {t̂}, {â}; not JSON braces (those are followed by a newline / quote).
_PLACEHOLDER = re.compile(r"\{([^{}\s\"'<>]+)\}")
_STEP_HEADER = re.compile(r"^#{2,4}\s*Step\s+(\d+)\b[^\n]*$", re.MULTILINE)
_TRANSITION_HEADER = re.compile(
    r"^#{2,4}\s*Step\s+(\d+)\s*(?:→|->|=>|—>|to)\s*Step\s+(\d+)[^\n]*$", re.MULTILINE
)


def render(template: str, values: Dict[str, Any]) -> str:
    """Fill ``{name}`` placeholders that appear in ``values``; leave everything else
    (e.g. literal JSON braces in OUTPUT FORMAT blocks) untouched."""
    def sub(m: re.Match) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)
    return _PLACEHOLDER.sub(sub, template)


def placeholders_in(template: str) -> set:
    return set(_PLACEHOLDER.findall(template))


def split_sections(text: str, names: List[str]) -> Dict[str, str]:
    """Split sectioned output. Missing sections map to ''.

    The prompts ask for ``===NAME===``, but models also write ``## NAME``,
    ``### NAME``, ``**NAME**`` or ``NAME:``; all are accepted, but only for the
    expected section names so ordinary headings are not mistaken for sections.
    """
    out = {n: "" for n in names}
    alt = "|".join(re.escape(n) for n in names)
    strict = re.compile(
        rf"^[ \t]*(?:=+|#+|\*\*)?[ \t]*({alt})[ \t]*(?:=+|\*\*)?[ \t]*:?[ \t]*(?:\*\*)?[ \t]*$",
        re.MULTILINE,
    )
    found = {m.start(): (m.group(1), m.start(), m.end()) for m in strict.finditer(text)}
    # Paraphrased headings, e.g. "## Memory Document 1 — High-level summary" or
    # "### HIGH-LEVEL MEMORY DOCUMENT" (seen with reasoning enabled). Only lines that
    # are headings (#, ===, **) are considered, so body text is never split.
    for name in names:
        loose = re.compile(
            r"^[ \t]*(?:#{1,6}|={2,}|\*\*)[^\n]*?\b" + name.replace("_", r"[-_ ]?") + r"\b[^\n]*$",
            re.MULTILINE | re.IGNORECASE,
        )
        for m in loose.finditer(text):
            found.setdefault(m.start(), (name, m.start(), m.end()))
    marks = [found[k] for k in sorted(found)]
    for i, (name, _start, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        if name in out:
            out[name] = text[end:stop].strip()
    return out


def parse_step_entries(section: str) -> Dict[int, str]:
    """``### Step N (Agent)`` entries -> {N: full entry text (header included)}."""
    entries: Dict[int, str] = {}
    heads = list(_STEP_HEADER.finditer(section))
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(section)
        n = int(m.group(1))
        body = section[m.start():stop].strip()
        entries[n] = (entries[n] + "\n" + body) if n in entries else body
    return entries


def parse_transitions(text: str) -> List[Dict[str, Any]]:
    """A.2 output -> [{'from': X, 'to': Y, 'label': ..., 'relationship': ...}]."""
    out: List[Dict[str, Any]] = []
    heads = list(_TRANSITION_HEADER.finditer(text))
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():stop]
        # Stop at the next level-2 section (e.g. "## Candidate-Pruning Summary").
        body = re.split(r"^##\s+\S", body, maxsplit=1, flags=re.MULTILINE)[0]
        label = None
        lm = re.search(r"Type\s*:\s*\**\s*([a-z_]+)", body, re.IGNORECASE)
        if lm and lm.group(1).lower() in TRANSITION_LABELS:
            label = lm.group(1).lower()
        rm = re.search(r"Relationship\s*:\s*(.+)", body, re.IGNORECASE)
        out.append({
            "from": int(m.group(1)),
            "to": int(m.group(2)),
            "label": label,
            "relationship": rm.group(1).strip() if rm else "",
        })
    return out


def transitions_involving(transitions: List[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
    return [t for t in transitions if t["from"] == step or t["to"] == step]


def format_transitions(transitions: List[Dict[str, Any]]) -> str:
    if not transitions:
        return "(none)"
    return "\n".join(
        f"- Step {t['from']} → Step {t['to']}: {t['label'] or 'unlabeled'} — {t['relationship']}"
        for t in transitions
    )


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the first parseable top-level JSON object in ``text`` (handles ```json fences)."""
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for blob in fenced:
        obj = _loads(blob)
        if obj is not None:
            return obj
    for start, end in _balanced_regions(text):
        obj = _loads(text[start:end])
        if obj is not None:
            return obj
    return None


def _loads(blob: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))  # trailing commas
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _balanced_regions(text: str) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                regions.append((start, i + 1))
    # Largest object first: the answer object usually encloses the others.
    return sorted(regions, key=lambda r: r[0] - r[1])


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group(0))
    return None


def contiguous_regions(steps: List[int]) -> List[List[int]]:
    """Merge adjacent indices into maximal intervals (paper §2.1 consolidation R)."""
    regions: List[List[int]] = []
    for n in sorted(set(steps)):
        if regions and n == regions[-1][-1] + 1:
            regions[-1].append(n)
        else:
            regions.append([n])
    return regions
