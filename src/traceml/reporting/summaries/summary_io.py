"""
Shared file and JSON helpers for end-of-run summary modules.

This module centralizes the small persistence helpers used by multiple summary
builders so the individual summary files can stay focused on aggregation and
presentation logic.
"""

import json
from typing import Any, Dict


def append_text(path: str, text: str) -> None:
    """
    Append text to a file, inserting a blank line first if the file already
    contains content.
    """
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0, 2)
        if f.tell() > 0:
            f.write("\n")
        f.write(text.rstrip() + "\n")


def load_json_or_empty(path: str) -> Dict[str, Any]:
    """
    Load a JSON file if it exists and is readable; otherwise return an empty
    object.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """
    Write JSON with stable indentation for human readability.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
