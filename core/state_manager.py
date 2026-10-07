"""Save / load experiments as self-contained TOML files (step 6).

File layout
    [experiment]   name, notes, app, format_version, saved
    [settings]     every SimSettings field (physics rule, grid, dt, ...)
    [[entity]]     one table per item on the grid (kind, x, y, z, vx, vy, vz, mass, ...)

Reading uses Python's built-in tomllib (3.11+). Writing uses the small writer below, so no extra
package is needed on Windows, Mac or Linux.
"""

from __future__ import annotations

import datetime as _dt
import math
import tomllib
from pathlib import Path

from core.entities import Entity

APP_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = APP_ROOT / "experiments"
FORMAT_VERSION = 1


# ----------------------------------------------------------------------------- tiny TOML writer
def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        esc = (v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                .replace("\r", "\\r").replace("\t", "\\t"))
        return f'"{esc}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if v is None:
        return '""'
    # numpy scalars and similar
    if hasattr(v, "item"):
        return _value(v.item())
    raise TypeError(f"Cannot write {type(v).__name__} to TOML")


def _key(k):
    k = str(k)
    return k if k.replace("_", "").replace("-", "").isalnum() else _value(k)


def toml_dumps(data: dict, header_comment: str = "") -> str:
    """Write a dict of {table: {key: value}} and {array_table: [ {..}, {..} ]} as TOML."""
    out = []
    if header_comment:
        out += [f"# {line}" for line in header_comment.splitlines()] + [""]
    plain = {k: v for k, v in data.items() if not isinstance(v, (dict, list)) or
             (isinstance(v, list) and not (v and isinstance(v[0], dict)))}
    for k, v in plain.items():
        out.append(f"{_key(k)} = {_value(v)}")
    if plain:
        out.append("")
    for k, v in data.items():
        if isinstance(v, dict):
            out.append(f"[{_key(k)}]")
            for kk, vv in v.items():
                out.append(f"{_key(kk)} = {_value(vv)}")
            out.append("")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for row in v:
                out.append(f"[[{_key(k)}]]")
                for kk, vv in row.items():
                    out.append(f"{_key(kk)} = {_value(vv)}")
                out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- experiments
def experiment_dict(settings: dict, entities: list, name: str = "", notes: str = "") -> dict:
    return {
        "experiment": {
            "name": name or "Untitled experiment",
            "notes": notes or "",
            "app": "TEGR Genesis",
            "format_version": FORMAT_VERSION,
            "saved": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "settings": dict(settings),
        "entity": [e.to_dict() if isinstance(e, Entity) else dict(e) for e in entities],
    }


def save_experiment(path, settings: dict, entities: list, name: str = "", notes: str = "") -> Path:
    path = Path(path)
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = toml_dumps(experiment_dict(settings, entities, name or path.stem, notes),
                      "TEGR Genesis experiment. Open it with File > Open Experiment.\n"
                      "Self-contained: physics settings + every item on the grid.")
    path.write_text(text, encoding="utf-8")
    return path


def load_experiment(path):
    """Return (settings dict, list of Entity, meta dict). Raises ValueError with a plain message."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"This file is not valid TOML:\n{exc}") from exc
    if "settings" not in data and "entity" not in data:
        raise ValueError("This TOML file is not a TEGR Genesis experiment\n"
                         "(it has no [settings] or [[entity]] sections).")
    meta = dict(data.get("experiment", {}))
    meta.setdefault("name", path.stem)
    settings = dict(data.get("settings", {}))
    entities = []
    for i, row in enumerate(data.get("entity", []), 1):
        try:
            entities.append(Entity.from_dict(row))
        except Exception as exc:
            raise ValueError(f"Item {i} in the file could not be read: {exc}") from exc
    return settings, entities, meta
