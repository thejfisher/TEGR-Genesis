"""Things you can place on the grid.

Kinds
  well    : Gravity Well   — fixed field source (no particle forces)
  defect  : Wave Defect    — a starting bump in the field (released at t = 0)
  anchor  : Static Anchor  — fixed particle, feeds the field, takes part in Pauli push
  active  : Active Body    — moving particle, feeds the field
  tracer  : Tracer         — moving particle, feels the field, does NOT feed it

All positions are in grid units, centred on the box (0, 0, 0). The setup grid is the z = 0 plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
import itertools

KINDS = ("well", "defect", "anchor", "active", "tracer")
FIXED_KINDS = ("well", "anchor")          # never move
SOURCE_KINDS = ("well", "anchor", "active")  # feed the field every tick
MOVING_KINDS = ("active", "tracer")
PAULI_KINDS = ("anchor", "active")        # take part in the particle-particle push

NICE_NAMES = {
    "well": "Gravity Well",
    "defect": "Wave Defect",
    "anchor": "Static Anchor",
    "active": "Active Body",
    "tracer": "Tracer",
}

_ids = itertools.count(1)


@dataclass
class Entity:
    kind: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    mass: float = 1.0          # source size (wells/anchors/active) and inertia (moving)
    amplitude: float = 1.0     # wave defects: peak height of the bump
    width: float = 6.0         # wave defects: bump width (grid units)
    phase: float = 0.0         # particle phase θ (used by the Pauli push)
    label: str = ""
    uid: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown kind {self.kind!r}; expected one of {KINDS}")
        for name in ("x", "y", "z", "vx", "vy", "vz", "mass", "amplitude", "width", "phase"):
            setattr(self, name, float(getattr(self, name)))
        if not self.label:
            self.label = f"{NICE_NAMES[self.kind]} {self.uid}"

    @property
    def is_particle(self):
        return self.kind in ("anchor", "active", "tracer")

    def to_dict(self):
        d = asdict(self)
        d.pop("uid", None)
        return d

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)} - {"uid"}
        return cls(**{k: v for k, v in d.items() if k in names})
