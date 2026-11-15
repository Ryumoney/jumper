"""A two-storey wooden terrace house, 3DK, about 63 m^2, at full scale.

**This scene adds a physical prop**: the walls, floors, stairs and built-in
fixtures collide, so the robot is shut inside the rooms and moves between them
through the doorways. It defines a space and nothing else -- no task, no reward.

## The plan

Drawn from a rental floor plan on the 910 mm module (半間) that Japanese wooden
houses are framed on. Both floors are 5 x 7.5 modules, 4.55 m east-west by
6.825 m north-south, so 31.1 m^2 a floor and 62.1 m^2 in all against the 63.35
the listing gives. The west wall is a party wall (a terrace house shares it with
the next unit), so it has no windows. South is -y.

    1F (y up = north)                       2F
    +---------+---------------+             +-------------------------+
    | bath    |               |             |                         |
    +---------+  DK  6.5 jo   |             |     Western room 7.5 jo |
    | washroom|  (kitchen N)  |             |                         |
    +----+----+               |             +----+------+-----+-------+
    | WC |hall+---- open -----+             |land|hall  | WC  | closet|
    +----+    |               |             |stair      +-----+-------+
    |stair    |  Japanese     |             |well|                    |
    |    |    |  room 5.5 jo  |             +----+  Western room 6.5  |
    +----+----+  (tatami)     |             |clo.|                    |
    |stor|entr|               |             +----+--------------------+
    +----+----+---------------+                       balcony 2.65 x 1.15

Room areas come out within half a jo of the listing: DK 2.73 x 3.64 m, the
Japanese room 2.73 x 3.18, the 7.5 jo room 4.55 x 2.73, the 6.5 jo room
3.64 x 2.73.

## The robot spawns in the DK

The prop is placed so the centre of the dining kitchen sits at every
environment's origin, which is where tasks spawn the robot (within +-0.5 m and
any yaw for `jumper.posture`, which `play --app` runs) -- clear of the kitchen
counter along the north wall.

## Three views, because a house hides what is inside it

A full-height house seen from the viewer's orbit camera is a roof. `--house-view`
chooses what is built:

- ``1f`` (default): the ground floor only, walls cut at 1.0 m like an
  architect's plan -- every room visible from above, and still far taller than
  the robot (0.11 m), so it cannot climb out.
- ``2f``: the ground floor whole and the first floor cut at 1.0 m.
- ``full``: both floors at full height, with the gable roof -- the outside.

The stairs are real (13 risers of 0.21 m), and so out of reach of a robot that
stands 0.11 m tall; they are part of the space, not a route.

Everything here is MuJoCo's builtin textures and primitive geoms plus one convex
mesh for the attic, so the scene is this one file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from .registry import GROUND, Headlight, Scene

# ── Dimensions (metres) ─────────────────────────────────────────────────────
MODULE = 0.91
W = 5 * MODULE            # east-west
D = 7.5 * MODULE          # north-south
EXT_T = 0.12              # exterior wall
INT_T = 0.08              # interior partition
FLOOR = 0.003             # finished floor above the ground plane (avoids z-fighting)
H1 = 2.4                  # ground-floor ceiling
F2 = 2.7                  # first-floor finished level
H2 = 2.4                  # first-floor ceiling
CUT = 1.0                 # where a cut-away floor's walls stop
DOOR_H = 2.0              # interior door head
PITCH = 0.4               # roof rise per run (4 sun)
EAVE = 0.45               # roof overhang

# Plan lines.
X_STRIP = 2 * MODULE      # 1.82: the bath/wash/hall strip | DK and Japanese room
X_STAIR = MODULE          # 0.91: the stair and WC column
Y_DK = 3.18               # Japanese room | DK
Y_WASH = 5 * MODULE       # 4.55: hall and WC | washroom
Y_BATH = 6 * MODULE       # 5.46: washroom | bath
Y_STAIR0 = 1.2            # foot of the stair
Y_STAIR1 = 4 * MODULE     # 3.64: head of the stair
Y_2F_BAND0 = 3 * MODULE   # 2.73: 6.5 jo room | the 2F middle band
Y_2F_BAND1 = 4.5 * MODULE  # 4.095: middle band | 7.5 jo room

#: Where the robot spawns, in house coordinates: the middle of the DK.
SPAWN = ((X_STRIP + W) / 2.0, 5.0)

VIEWS = ("1f", "2f", "full")
VIEW_ENV = "MJRL_HOUSE_VIEW"

# ── Materials ───────────────────────────────────────────────────────────────
# Each entry: texture builtin, colours, texrepeat per metre. A checker's cell is
# 1 / (2 * texrepeat) across, so planks and tatami are sized through texrepeat.


@dataclass(frozen=True)
class _Mat:
    builtin: str                       # "checker" | "flat" | "none"
    rgb1: tuple[float, float, float]
    rgb2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mark: str = "none"                 # "edge" | "random" | "none"
    markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    random: float = 0.0
    repeat: tuple[float, float] = (1.0, 1.0)
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    specular: float = 0.1
    emission: float = 0.0
    extra: dict = field(default_factory=dict)


_MATS: dict[str, _Mat] = {
    # Oak boards 0.15 m wide and 0.9 m long, running east-west.
    "oak": _Mat("checker", (0.69, 0.52, 0.34), (0.63, 0.46, 0.29), "edge", (0.42, 0.30, 0.19),
                repeat=(0.55, 3.3)),
    # Tatami: 0.91 x 1.82 mats with their dark cloth border.
    "tatami": _Mat("checker", (0.70, 0.68, 0.46), (0.66, 0.65, 0.42), "edge", (0.22, 0.24, 0.16),
                   repeat=(0.55, 0.275)),
    "tile": _Mat("checker", (0.88, 0.88, 0.86), (0.84, 0.84, 0.82), "edge", (0.62, 0.62, 0.60),
                 repeat=(5.0, 5.0), specular=0.3),
    "genkan": _Mat("checker", (0.46, 0.44, 0.41), (0.42, 0.40, 0.37), "edge", (0.30, 0.29, 0.27),
                   repeat=(1.67, 1.67)),
    # Warm plaster for the inside faces.
    "plaster": _Mat("flat", (0.92, 0.89, 0.83), (0.92, 0.89, 0.83), "random", (0.86, 0.83, 0.77),
                    random=0.25, repeat=(2.0, 2.0)),
    # Cedar weatherboard outside: courses 0.18 m high.
    "siding": _Mat("checker", (0.55, 0.37, 0.22), (0.50, 0.33, 0.19), "edge", (0.30, 0.19, 0.11),
                   repeat=(0.25, 2.8)),
    "timber": _Mat("flat", (0.40, 0.27, 0.16), (0.40, 0.27, 0.16), "random", (0.33, 0.22, 0.13),
                   random=0.3, repeat=(4.0, 4.0)),
    "stair": _Mat("flat", (0.60, 0.44, 0.27), (0.60, 0.44, 0.27), "random", (0.52, 0.37, 0.22),
                  random=0.3, repeat=(4.0, 4.0)),
    "roof": _Mat("checker", (0.24, 0.25, 0.27), (0.21, 0.22, 0.24), "edge", (0.14, 0.14, 0.15),
                 repeat=(2.5, 2.5)),
    "white": _Mat("none", (0, 0, 0), rgba=(0.95, 0.95, 0.93, 1.0), specular=0.4),
    "steel": _Mat("none", (0, 0, 0), rgba=(0.72, 0.73, 0.74, 1.0), specular=0.6),
    "cabinet": _Mat("none", (0, 0, 0), rgba=(0.86, 0.82, 0.74, 1.0), specular=0.2),
    "glass": _Mat("none", (0, 0, 0), rgba=(0.70, 0.85, 0.95, 0.22), specular=0.8),
    "stone": _Mat("flat", (0.55, 0.54, 0.51), (0.55, 0.54, 0.51), "random", (0.45, 0.44, 0.42),
                  random=0.4, repeat=(6.0, 6.0)),
    "lamp": _Mat("none", (0, 0, 0), rgba=(1.0, 0.92, 0.75, 1.0), emission=0.8),
}


def _add_materials(spec) -> None:
    import mujoco

    builtins = {"checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT}
    marks = {"edge": mujoco.mjtMark.mjMARK_EDGE, "random": mujoco.mjtMark.mjMARK_RANDOM,
             "none": mujoco.mjtMark.mjMARK_NONE}
    for name, m in _MATS.items():
        material = spec.add_material(
            name=name, rgba=list(m.rgba), specular=m.specular, shininess=0.2,
            emission=m.emission, texrepeat=list(m.repeat), texuniform=True,
        )
        if m.builtin == "none":
            continue
        spec.add_texture(
            name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=builtins[m.builtin], rgb1=list(m.rgb1), rgb2=list(m.rgb2),
            mark=marks[m.mark], markrgb=list(m.markrgb), random=m.random,
            width=256, height=256,
        )
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"


# ── The builder ─────────────────────────────────────────────────────────────


class _House:
    """Adds geoms to one body. Coordinates are house coordinates: the south-west
    corner of the footprint at the ground, x east, y north, z up."""

    def __init__(self, spec, body) -> None:
        import mujoco

        self.mj = mujoco
        self.spec = spec
        self.body = body

    # Primitive adders. `solid=False` is decoration: it neither collides nor,
    # on the native backend, survives model slimming -- it is only drawn.
    def box(self, lo, hi, material, solid=True) -> None:
        size = [(b - a) / 2.0 for a, b in zip(lo, hi, strict=True)]
        if min(size) <= 1e-6:
            return
        pos = [(b + a) / 2.0 for a, b in zip(lo, hi, strict=True)]
        self._geom(self.mj.mjtGeom.mjGEOM_BOX, size, pos, material, solid)

    def cyl(self, x, y, z0, z1, r, material, solid=True) -> None:
        self._geom(self.mj.mjtGeom.mjGEOM_CYLINDER, [r, (z1 - z0) / 2.0],
                   [x, y, (z0 + z1) / 2.0], material, solid)

    def _geom(self, gtype, size, pos, material, solid, quat=None):
        # Group 0, with the ground: the viewer hides a group whose geoms all
        # collide when their body also owns drawn-only geoms
        # (`LiveViewer._collision_only_groups`), which would make every wall
        # vanish from the picture.
        geom = self.body.add_geom(type=gtype, size=list(size), pos=list(pos),
                                  material=material, group=0)
        if not solid:
            geom.contype = 0
            geom.conaffinity = 0
        if quat is not None:
            geom.quat = list(quat)
        return geom

    # Walls.
    def wall(self, axis: str, at: float, lo: float, hi: float, z0: float, z1: float,
             t: float, openings=(), outside: int = 0) -> None:
        """A straight wall with openings.

        Args:
            axis: "x" for a wall running east-west at y = `at`, "y" for one
                running north-south at x = `at`.
            lo, hi: its extent along that axis.
            z0, z1: its base and top. A cut-away floor passes a lower `z1`.
            openings: (a, b, sill, head, kind) with kind "door" (an empty
                doorway), "window" (glass between sill and head), or "entry" (a
                closed wooden front door).
            outside: -1 or +1 for an exterior wall, naming the side that faces
                out (weatherboard goes there); 0 for a partition.
        """
        cur = lo
        for a, b, sill, head, kind in sorted(openings):
            self._piece(axis, at, cur, a, z0, z1, t, outside, z0)
            if sill > 0:
                self._piece(axis, at, a, b, z0, min(z0 + sill, z1), t, outside, z0)
            if z0 + head < z1:
                self._piece(axis, at, a, b, z0 + head, z1, t, outside, z0)
            g0, g1 = z0 + sill, min(z0 + head, z1)
            if g1 > g0 and kind == "window":
                self._slab(axis, at, a, b, g0, g1, 0.012, "glass", solid=True)
                self._frame(axis, at, a, b, g0, g1, t)
            elif g1 > g0 and kind == "entry":
                self._slab(axis, at, a, b, g0, g1, 0.045, "timber", solid=True)
                self._slab(axis, at, a + 0.15, b - 0.15, g0 + 0.9, min(g1, g0 + 1.7),
                           0.05, "glass", solid=False)
            elif g1 > g0:  # doorway: a timber lining, drawn only
                self._frame(axis, at, a, b, g0, g1, t)
            cur = b
        self._piece(axis, at, cur, hi, z0, z1, t, outside, z0)

    def _span(self, axis, at, s0, s1, z0, z1, t0, t1):
        """(lo, hi) corners of a box along the wall, t0..t1 across it."""
        if axis == "x":
            return (s0, at + t0, z0), (s1, at + t1, z1)
        return (at + t0, s0, z0), (at + t1, s1, z1)

    def _slab(self, axis, at, s0, s1, z0, z1, t, material, solid):
        if s1 > s0 and z1 > z0:
            self.box(*self._span(axis, at, s0, s1, z0, z1, -t / 2, t / 2), material, solid)

    def _piece(self, axis, at, s0, s1, z0, z1, t, outside, floor_z):
        if s1 - s0 <= 1e-6 or z1 - z0 <= 1e-6:
            return
        self.box(*self._span(axis, at, s0, s1, z0, z1, -t / 2, t / 2), "plaster")
        if outside:
            o = outside * (t / 2 + 0.012)
            self.box(*self._span(axis, at, s0, s1, z0, z1, o - 0.012, o + 0.012),
                     "siding", solid=False)
        if abs(z0 - floor_z) < 1e-9:  # skirting board on each inside face
            for side in (-1, 1):
                if side == outside:
                    continue
                o = side * (t / 2 + 0.006)
                self.box(*self._span(axis, at, s0, s1, z0, min(z0 + 0.07, z1),
                                     o - 0.006, o + 0.006), "timber", solid=False)

    def _frame(self, axis, at, a, b, z0, z1, t):
        w, d = 0.035, t / 2 + 0.01
        self.box(*self._span(axis, at, a - w, a, z0, z1, -d, d), "timber", solid=False)
        self.box(*self._span(axis, at, b, b + w, z0, z1, -d, d), "timber", solid=False)

    def post(self, x, y, z0, z1) -> None:
        self.box((x - 0.075, y - 0.075, z0), (x + 0.075, y + 0.075, z1), "timber", solid=False)

    def floor(self, x0, y0, x1, y1, level, material) -> None:
        self.box((x0, y0, level - 0.05), (x1, y1, level), material)

    def light(self, x, y, z) -> None:
        self.body.add_light(pos=[x, y, z], dir=[0, 0, -1], diffuse=[0.35, 0.32, 0.27],
                            specular=[0.0, 0.0, 0.0], attenuation=[1.0, 0.0, 0.08],
                            castshadow=False)
        self.cyl(x, y, z - 0.02, z, 0.18, "lamp", solid=False)


# ── The house ───────────────────────────────────────────────────────────────


def _ground_floor(h: _House, top_ext: float, top_int: float) -> None:
    z0, cut = 0.0, top_int
    e, i = EXT_T, INT_T
    win = "window"

    # Floors.
    h.floor(X_STRIP, Y_DK, W, D, FLOOR, "oak")                    # DK
    h.floor(X_STRIP, 0.0, W, Y_DK, FLOOR, "tatami")               # Japanese room
    h.floor(X_STAIR, 1.0, X_STRIP, Y_WASH, FLOOR, "oak")          # hall
    h.floor(X_STAIR, 0.0, X_STRIP, 1.0, FLOOR, "genkan")          # entrance
    h.floor(0.0, 0.0, X_STAIR, Y_STAIR1, FLOOR, "oak")            # store and stair
    h.floor(0.0, Y_STAIR1, X_STAIR, Y_WASH, FLOOR, "tile")        # WC
    h.floor(0.0, Y_WASH, X_STRIP, D, FLOOR, "tile")               # washroom and bath

    # Exterior walls.
    h.wall("x", 0.0, -e / 2, W + e / 2, z0, top_ext, e, outside=-1, openings=[
        (0.98, 1.74, 0.0, 2.0, "entry"),
        (1.97, 4.43, 0.0, 2.0, win),          # three sliding glass doors
    ])
    h.wall("x", D, -e / 2, W + e / 2, z0, top_ext, e, outside=1, openings=[
        (0.40, 0.95, 1.3, 1.85, win),          # bath
        (3.00, 4.15, 1.1, 1.9, win),           # over the kitchen
    ])
    h.wall("y", 0.0, -e / 2, D + e / 2, z0, top_ext, e, outside=-1)   # party wall
    h.wall("y", W, -e / 2, D + e / 2, z0, top_ext, e, outside=1, openings=[
        (0.80, 2.40, 0.9, 1.9, win),           # Japanese room
        (3.90, 5.60, 0.9, 1.9, win),           # DK
    ])

    # Partitions.
    h.wall("y", X_STRIP, 0.0, D, z0, cut, i, openings=[
        (1.35, 2.10, 0.0, DOOR_H, "door"),     # hall -> Japanese room
        (3.75, 4.45, 0.0, DOOR_H, "door"),     # hall -> DK
    ])
    h.wall("x", Y_DK, X_STRIP, W, z0, cut, i, openings=[
        (1.95, 3.55, 0.0, DOOR_H, "door"),     # DK <-> Japanese room, doors taken out
    ])
    # The Japanese room's closet, open-fronted.
    h.wall("x", Y_DK - MODULE, 4 * MODULE, W, z0, cut, i, openings=[
        (3.72, 4.47, 0.0, 1.8, "door"),
    ])
    h.wall("y", 4 * MODULE, Y_DK - MODULE, Y_DK, z0, cut, i)
    h.box((4 * MODULE, Y_DK - MODULE, 0.78), (W, Y_DK, 0.82), "timber")  # its shelf
    # Store | entrance, and WC | hall. The stair between them is open to the hall.
    h.wall("y", X_STAIR, 0.0, Y_STAIR0, z0, cut, i, openings=[(0.20, 0.95, 0.0, 1.8, "door")])
    h.wall("y", X_STAIR, Y_STAIR1, Y_WASH, z0, cut, i, openings=[(3.72, 4.42, 0.0, DOOR_H, "door")])
    h.wall("x", Y_STAIR0, 0.0, X_STAIR, z0, cut, i)
    h.wall("x", Y_STAIR1, 0.0, X_STAIR, z0, cut, i)
    h.wall("x", Y_WASH, 0.0, X_STRIP, z0, cut, i, openings=[(0.98, 1.74, 0.0, DOOR_H, "door")])
    h.wall("x", Y_BATH, 0.0, X_STRIP, z0, cut, i, openings=[(0.95, 1.70, 0.0, DOOR_H, "door")])

    # The stair: 13 risers to the first floor, 12 treads on the ground floor.
    risers = 13
    rise = F2 / risers
    tread = (Y_STAIR1 - Y_STAIR0) / (risers - 1)
    for k in range(risers - 1):
        top = (k + 1) * rise
        if top > cut + 1e-9:
            break
        h.box((0.06, Y_STAIR0 + k * tread, 0.0),
              (X_STAIR - 0.04, Y_STAIR0 + (k + 1) * tread, top), "stair")

    # Built-in fixtures.
    wall_n = D - e / 2
    h.box((2.60, wall_n - 0.65, 0.0), (W - e / 2, wall_n, 0.82), "cabinet")      # kitchen
    h.box((2.60, wall_n - 0.65, 0.82), (W - e / 2, wall_n, 0.85), "steel")
    h.box((3.05, wall_n - 0.50, 0.85), (3.65, wall_n - 0.12, 0.86), "white", solid=False)
    h.box((3.90, wall_n - 0.55, 0.85), (4.40, wall_n - 0.10, 0.87), "timber", solid=False)
    h.box((0.06, D - e / 2 - 0.75, 0.0), (1.30, D - e / 2, 0.55), "white")       # bath tub
    h.box((0.06, 4.70, 0.0), (0.62, 5.40, 0.80), "cabinet")                       # vanity
    h.box((1.10, 4.62, 0.0), (1.74, 5.26, 0.05), "white")                         # washer tray
    _toilet(h, 0.06, (Y_STAIR1 + Y_WASH) / 2, 0.0, facing=1)

    # Exposed corner posts and two ceiling lights.
    for x, y in ((0, 0), (W, 0), (0, D), (W, D)):
        h.post(x, y, 0.0, top_ext)
    if top_int >= H1:
        h.light(SPAWN[0], SPAWN[1], H1 - 0.02)
        h.light((X_STRIP + W) / 2, Y_DK / 2, H1 - 0.02)


def _first_floor(h: _House, top_ext: float, top_int: float) -> None:
    z0, e, i = F2, EXT_T, INT_T
    win = "window"

    # The slab between the floors, open over the stair, and the finished floors.
    h.box((0.0, Y_STAIR1, H1), (W, D, F2 - 0.05), "plaster")
    h.box((X_STAIR, 0.0, H1), (W, Y_STAIR1, F2 - 0.05), "plaster")
    h.box((0.0, 0.0, H1), (X_STAIR, Y_STAIR0, F2 - 0.05), "plaster")
    h.floor(0.0, Y_2F_BAND1, W, D, F2 + FLOOR, "oak")              # 7.5 jo
    h.floor(X_STAIR, 0.0, W, Y_2F_BAND0, F2 + FLOOR, "oak")        # 6.5 jo
    h.floor(X_STAIR, Y_2F_BAND0, 2.0, Y_2F_BAND1, F2 + FLOOR, "oak")  # hall
    h.floor(0.0, Y_STAIR1, X_STAIR, Y_2F_BAND1, F2 + FLOOR, "oak")  # landing
    h.floor(2.0, Y_2F_BAND0, 2.91, Y_2F_BAND1, F2 + FLOOR, "tile")  # WC
    h.floor(2.91, Y_2F_BAND0, W, Y_2F_BAND1, F2 + FLOOR, "oak")    # closet
    h.floor(0.0, 0.0, X_STAIR, Y_STAIR0, F2 + FLOOR, "oak")        # closet

    # Exterior walls.
    h.wall("x", 0.0, -e / 2, W + e / 2, z0, top_ext, e, outside=-1, openings=[
        (1.97, 4.43, 0.0, 2.0, win),           # to the balcony
    ])
    h.wall("x", D, -e / 2, W + e / 2, z0, top_ext, e, outside=1, openings=[
        (0.60, 2.00, 0.9, 1.9, win), (2.60, 4.00, 0.9, 1.9, win),
    ])
    h.wall("y", 0.0, -e / 2, D + e / 2, z0, top_ext, e, outside=-1)
    h.wall("y", W, -e / 2, D + e / 2, z0, top_ext, e, outside=1, openings=[
        (0.60, 2.20, 0.9, 1.9, win), (4.60, 6.20, 0.9, 1.9, win),
    ])

    # Partitions.
    h.wall("x", Y_2F_BAND1, 0.0, W, z0, top_int, i, openings=[
        (1.00, 1.75, 0.0, DOOR_H, "door"),     # hall -> 7.5 jo
        (3.10, 4.45, 0.0, 1.8, "door"),        # 7.5 jo -> its closet
    ])
    h.wall("x", Y_2F_BAND0, X_STAIR, W, z0, top_int, i, openings=[
        (1.00, 1.75, 0.0, DOOR_H, "door"),     # hall -> 6.5 jo
    ])
    h.wall("y", X_STAIR, 0.0, Y_2F_BAND0, z0, top_int, i, openings=[
        (0.20, 1.00, 0.0, 1.8, "door"),        # 6.5 jo -> closet over the store
    ])
    h.wall("x", Y_STAIR0, 0.0, X_STAIR, z0, top_int, i)
    h.wall("y", 2.0, Y_2F_BAND0, Y_2F_BAND1, z0, top_int, i, openings=[
        (2.90, 3.60, 0.0, DOOR_H, "door"),
    ])
    h.wall("y", 2.91, Y_2F_BAND0, Y_2F_BAND1, z0, top_int, i)
    # Guard rail over the stairwell, on the hall side.
    rail_top = min(F2 + 0.85, top_int)
    h.box((X_STAIR - 0.02, Y_2F_BAND0, F2 + 0.80), (X_STAIR + 0.02, Y_STAIR1, rail_top + 0.05),
          "timber")
    for y in (Y_2F_BAND0 + 0.05, (Y_2F_BAND0 + Y_STAIR1) / 2, Y_STAIR1 - 0.05):
        h.box((X_STAIR - 0.02, y - 0.02, F2), (X_STAIR + 0.02, y + 0.02, F2 + 0.85), "timber")
    _toilet(h, (2.0 + 2.91) / 2, Y_2F_BAND1 - i / 2, F2, facing=-2)

    for x, y in ((0, 0), (W, 0), (0, D), (W, D)):
        h.post(x, y, F2, top_ext)

    # Balcony: 2.65 x 1.15 timber deck on two posts, rail and a laundry pole.
    bx0, by0 = W - 2.65, -1.15
    h.box((bx0, by0, F2 - 0.12), (W, -EXT_T / 2, F2), "stair")
    for x in (bx0 + 0.06, W - 0.06):
        h.box((x - 0.05, by0 + 0.01, 0.0), (x + 0.05, by0 + 0.11, F2 - 0.12), "timber")
    for (x0, y0), (x1, y1) in (((bx0, by0), (W, by0 + 0.04)),
                               ((bx0, by0), (bx0 + 0.04, 0.0)),
                               ((W - 0.04, by0), (W, 0.0))):
        h.box((x0, y0, F2 + 0.95), (x1, y1, F2 + 1.0), "timber")
        n = max(2, int(max(x1 - x0, y1 - y0) / 0.12))
        for k in range(n + 1):
            f = k / n
            x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
            h.box((x - 0.015, y - 0.015, F2), (x + 0.015, y + 0.015, F2 + 0.95), "timber",
                  solid=False)
    # The laundry pole, lying east-west (a cylinder's axis is z; turn it onto x).
    h._geom(h.mj.mjtGeom.mjGEOM_CYLINDER, [0.015, 1.2], [(bx0 + W) / 2, by0 + 0.3, F2 + 1.6],
            "steel", solid=False, quat=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])


def _toilet(h: _House, x: float, y: float, z0: float, facing: int) -> None:
    """A WC against a wall. facing=1: tank on the west wall, bowl to the east;
    facing=-2: tank on the north wall, bowl to the south."""
    if facing == 1:
        h.box((x, y - 0.22, z0), (x + 0.20, y + 0.22, z0 + 0.80), "white")
        h.cyl(x + 0.42, y, z0, z0 + 0.40, 0.19, "white")
    else:
        h.box((x - 0.22, y - 0.20, z0), (x + 0.22, y, z0 + 0.80), "white")
        h.cyl(x, y - 0.42, z0, z0 + 0.40, 0.19, "white")


def _roof(h: _House, spec) -> None:
    """Attic as one convex prism, then the two pitched planes and the entrance canopy."""
    import mujoco

    z = F2 + H2
    ridge = z + (D / 2) * PITCH
    verts = []
    for x in (-EXT_T / 2, W + EXT_T / 2):
        verts += [x, -EXT_T / 2, z, x, D + EXT_T / 2, z, x, D / 2, ridge]
    spec.add_mesh(name="house_attic", uservert=verts)
    attic = h.body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="house_attic",
                            material="siding", group=0)
    attic.contype = attic.conaffinity = 0

    angle = math.atan(PITCH)
    run = D / 2 + EAVE
    half_len = run / math.cos(angle) / 2
    for side in (-1, 1):
        # Each plane runs from the ridge to its eave; place it at the midpoint of
        # that line, lifted by its half-thickness, and tilt it down outwards.
        y_mid = (D / 2 + (D + EAVE if side > 0 else -EAVE)) / 2
        z_mid = ridge - abs(y_mid - D / 2) * PITCH + 0.04
        a = -side * angle
        h._geom(mujoco.mjtGeom.mjGEOM_BOX, [W / 2 + EAVE, half_len, 0.04],
                [W / 2, y_mid, z_mid], "roof", solid=False,
                quat=[math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])
    # Entrance canopy.
    h.box((0.75, -0.65, 2.25), (1.95, -EXT_T / 2, 2.31), "roof", solid=False)


def _garden(h: _House) -> None:
    """Stepping stones from the front door south, across the lawn."""
    for k in range(4):
        h.cyl(1.36 + (0.08 if k % 2 else -0.06), -0.45 - 0.42 * k, 0.0, 0.006, 0.16,
              "stone", solid=False)


def _house_spec(view: str):
    """The house as an `MjSpec`: one fixed body, so mjlab mocap-wraps it and
    places one per environment (`registry._place_props` adds the reset event)."""
    import mujoco

    if view not in VIEWS:
        raise ValueError(f"house view {view!r} is not one of {VIEWS}")
    spec = mujoco.MjSpec()
    _add_materials(spec)
    body = spec.worldbody.add_body(name="house")
    h = _House(spec, body)

    if view == "1f":
        _ground_floor(h, top_ext=CUT, top_int=CUT)
    else:
        _ground_floor(h, top_ext=F2, top_int=H1)
        top = F2 + (CUT if view == "2f" else H2)
        _first_floor(h, top_ext=top, top_int=top)
        if view == "full":
            _roof(h, spec)
    _garden(h)
    return spec


def house_entity(view: str = "1f"):
    """The house as an mjlab entity, placed so the DK's centre is at the origin."""
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=lambda: _house_spec(view),
        init_state=EntityCfg.InitialStateCfg(
            # Relative to each environment's own origin.
            pos=(-SPAWN[0], -SPAWN[1], 0.0),
            joint_pos={".*": 0.0},
        ),
    )


def cli_args(group):
    """`--house-view`; see the module docstring. Returns the dest names it added."""
    group.add_argument(
        "--house-view", default=os.environ.get(VIEW_ENV, "1f"), choices=VIEWS,
        help="what of the house to build: 1f (ground floor, walls cut at 1 m so the "
        "rooms show from above; default), 2f (first floor cut instead), full "
        f"(both floors and the roof). Falls back to {VIEW_ENV} in .env",
    )
    return ["house_view"]


def scene(house_view: str = "1f") -> Scene:
    """Lawn outside, a soft afternoon sun, and the house."""
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.22, 0.21, 0.20), diffuse=(0.30, 0.29, 0.28)),
        textures=(
            sc.TextureCfg(
                name="house_lawn", type="2d", builtin="flat", mark="random",
                rgb1=(0.27, 0.38, 0.18), rgb2=(0.27, 0.38, 0.18),
                markrgb=(0.35, 0.47, 0.23), random=0.4, width=512, height=512,
            ),
            sc.TextureCfg(
                name="house_sky", type="skybox", builtin="gradient",
                rgb1=(0.45, 0.65, 0.88), rgb2=(0.94, 0.91, 0.84),
                width=512, height=512,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="house_lawn", texture="house_lawn", texuniform=True,
                texrepeat=(6.0, 6.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(
                name="sun", type="directional", pos=(0.0, 0.0, 8.0),
                dir=(-0.35, 0.45, -0.82), castshadow=True,
                diffuse=(0.62, 0.58, 0.52), specular=(0.15, 0.15, 0.15),
                ambient=(0.10, 0.10, 0.11),
            ),
        ),
        props={"house": house_entity(house_view)},
    )
