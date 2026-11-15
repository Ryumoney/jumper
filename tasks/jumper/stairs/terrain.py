"""A straight staircase as an mjlab sub-terrain: flat run-up, N steps, landing.

**Why a sub-terrain and not a prop entity.** Everything in the velocity skeleton
that reads the ground -- `feet_ground_contact` (secondary body ``terrain``), the
foot height scan, the critic's `terrain_scan` -- is wired to the generator's
``terrain`` body. A staircase added as a separate entity would be stood on but
not seen as ground: air time, stance load and foot clearance would all read the
robot as airborne the moment it reached the first tread, quietly.

Layout of one patch, in the patch's own frame (x along the stairs):

    0 ........ RUN_UP ......... RUN_UP + N*tread ............... size[0]
    | flat floor | step 1 | step 2 | ... | step N |   landing (top)   |
         ^ spawn (origin)

The patch spans its whole width with the staircase, so a robot drifting
sideways steps onto the neighbouring column, which is the same difficulty.

## The garden look (`garden=True`), replay only

A mossy stone stair in a garden: granite treads with moss at the foot of each
riser, a lawn run-up, a paved landing, stone lanterns and shrubs on the column
seams, maples at the far end, a sky and a sun. **It is for watching, and
`env_cfg` turns it on only under `play`**, for two reasons that are both silent:

- **Ray sensors see visual geoms.** `mj_ray` filters by geom group, not by
  `contype`, and mjlab's ray casts take groups 0-2 -- the same groups the viewer
  draws. A lantern with `contype=0` is still "ground" to the foot height scan
  and the critic's `terrain_scan`, so in training it would change rewards and
  observations near the seams while the physics stayed the same.
- **The generator overwrites every terrain geom's rgba** (`color_scheme`), to
  0.5 grey under ``"none"``. That grey is MuJoCo's default geom rgba, which is
  the one value that lets a geom's *material* colour show through -- so the look
  is carried entirely by materials, never by `rgba`. (It is also why the step
  colours this file used to compute were never seen.)

The physical geoms -- floor, steps, landing -- are the same boxes with or without
it; `garden` only gives them materials and adds non-colliding decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

_FLOOR_RGBA = (0.55, 0.55, 0.58, 1.0)
_STEP_RGBA_LOW = np.array((0.80, 0.62, 0.40, 1.0))
_STEP_RGBA_HIGH = np.array((0.55, 0.36, 0.20, 1.0))


@dataclass(kw_only=True)
class StraightStairsTerrainCfg(SubTerrainCfg):
    """Configuration for a straight staircase climbing along +x."""

    step_height_range: tuple[float, float]
    """Rise of every step at difficulty 0 and 1, in metres. Interpolated."""
    tread: float
    """Depth (run) of each step, in metres."""
    num_steps: int
    """How many steps between the floor and the landing."""
    run_up: float = 1.0
    """Flat floor before the first riser, in metres. The robot spawns at half of it."""
    garden: bool = False
    """Dress the patch as a mossy stone garden stair. For replay only -- see the
    module docstring for why it must stay off in training."""
    garden_margin: float = 0.0
    """How far the generator's border reaches beyond the patch on each side, in
    metres -- `TerrainGeneratorCfg.border_width`. The garden lays lawn over that
    much of it, beside the patch, where the border is otherwise bare dark grey.
    With the curriculum on there is one column, so this is all of the side
    ground there is."""

    def stairs_length(self) -> float:
        return self.num_steps * self.tread

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # Deterministic geometry.
        lo, hi = self.step_height_range
        rise = lo + difficulty * (hi - lo)
        length, width = self.size
        landing_x0 = self.run_up + self.stairs_length()
        if landing_x0 >= length:
            raise ValueError(
                f"patch length {length} m leaves no landing after the run-up "
                f"({self.run_up} m) and {self.num_steps} x {self.tread} m of stairs"
            )

        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []
        if self.garden:
            _garden_assets(spec)

        # Floor under the whole patch, top at z = 0.
        floor = make_plane(body, (length, width), 0.0, center_zero=False)[0]
        geometries.append(TerrainGeometry(geom=floor, color=_FLOOR_RGBA))
        if self.garden:
            floor.material = "garden_lawn"

        # Solid blocks from z = 0 up to each step's top, so no riser is hollow.
        for k in range(self.num_steps):
            top = (k + 1) * rise
            if top <= 0.0:
                continue
            x0 = self.run_up + k * self.tread
            t = k / max(self.num_steps - 1, 1)
            rgba = tuple(_STEP_RGBA_LOW + t * (_STEP_RGBA_HIGH - _STEP_RGBA_LOW))
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(self.tread / 2.0, width / 2.0, top / 2.0),
                pos=(x0 + self.tread / 2.0, width / 2.0, top / 2.0),
            )
            geometries.append(TerrainGeometry(geom=geom, color=rgba))
            if self.garden:
                geom.material = "garden_stone_a" if k % 2 == 0 else "garden_stone_b"

        # Landing: from the last riser to the end of the patch, at the top height.
        top = self.num_steps * rise
        if top > 0.0:
            landing_len = length - landing_x0
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(landing_len / 2.0, width / 2.0, top / 2.0),
                pos=(landing_x0 + landing_len / 2.0, width / 2.0, top / 2.0),
            )
            geometries.append(TerrainGeometry(geom=geom, color=tuple(_STEP_RGBA_HIGH)))
            if self.garden:
                geom.material = "garden_paving"

        if self.garden:
            geometries += _garden_decor(
                body, length=length, width=width, run_up=self.run_up,
                tread=self.tread, num_steps=self.num_steps, rise=rise,
                margin=self.garden_margin, seed=round(difficulty * 1000),
            )

        origin = np.array((self.run_up / 2.0, width / 2.0, 0.0))
        return TerrainOutput(origin=origin, geometries=geometries)



# ── The garden look ─────────────────────────────────────────────────────────
# Sizes are set against the robot, which stands about 0.11 m tall and 0.33 m
# long: a lantern is four of it high, a maple about ten.
#
# Colours are kept dark on purpose. mjlab's scene template fixes the headlight at
# 0.3 ambient + 0.6 diffuse, and the parent spec's `<visual>` wins over anything a
# terrain sets, so the sun is added on top of a strong fill: a mid-green lawn
# rendered as lime and the granite as white before these were lowered.

#: name -> (rgba, emission, texture or None). Textures are procedural (no files).
_MATERIALS: dict[str, tuple[tuple[float, float, float, float], float, str | None]] = {
    "garden_lawn": ((1, 1, 1, 1), 0.0, "garden_lawn_tex"),
    "garden_stone_a": ((1, 1, 1, 1), 0.0, "garden_granite_tex"),
    "garden_stone_b": ((0.90, 0.90, 0.88, 1), 0.0, "garden_granite_tex"),
    "garden_paving": ((1, 1, 1, 1), 0.0, "garden_paving_tex"),
    "garden_wall": ((0.80, 0.78, 0.74, 1), 0.0, "garden_granite_tex"),
    "garden_moss": ((0.22, 0.33, 0.13, 1), 0.0, None),
    "garden_lantern": ((0.78, 0.77, 0.73, 1), 0.0, "garden_granite_tex"),
    "garden_lantern_light": ((1.0, 0.78, 0.40, 1), 0.8, None),
    "garden_shrub": ((0.12, 0.25, 0.10, 1), 0.0, None),
    "garden_shrub_light": ((0.20, 0.34, 0.13, 1), 0.0, None),
    "garden_flower": ((0.92, 0.62, 0.70, 1), 0.15, None),
    "garden_trunk": ((0.22, 0.16, 0.12, 1), 0.0, None),
    "garden_maple_red": ((0.62, 0.14, 0.08, 1), 0.0, None),
    "garden_maple_orange": ((0.78, 0.36, 0.08, 1), 0.0, None),
    "garden_gravel": ((1, 1, 1, 1), 0.0, "garden_gravel_tex"),
}


def _garden_assets(spec: mujoco.MjSpec) -> None:
    """Textures, materials, sky and sun -- once per spec, however many patches."""
    if spec.material("garden_lawn") is not None:
        return
    tex2d = mujoco.mjtTexture.mjTEXTURE_2D
    flat = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    speckle = mujoco.mjtMark.mjMARK_RANDOM

    def speckled(name: str, base, mark, amount: float) -> None:
        spec.add_texture(name=name, type=tex2d, builtin=flat, rgb1=base, rgb2=base,
                         mark=speckle, random=amount, markrgb=mark, width=256, height=256)

    speckled("garden_lawn_tex", [0.20, 0.30, 0.13], [0.30, 0.42, 0.18], 0.45)
    speckled("garden_granite_tex", [0.46, 0.45, 0.42], [0.32, 0.31, 0.30], 0.30)
    speckled("garden_gravel_tex", [0.50, 0.48, 0.44], [0.36, 0.34, 0.31], 0.50)
    spec.add_texture(
        name="garden_paving_tex", type=tex2d, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.48, 0.46, 0.42], rgb2=[0.45, 0.43, 0.40],
        mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.30, 0.29, 0.27],
        width=256, height=256,
    )
    spec.add_texture(
        name="garden_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.42, 0.62, 0.85], rgb2=[0.93, 0.89, 0.80], width=512, height=3072,
    )
    # texrepeat per metre (texuniform): fine grain on lawn and gravel, 0.25 m
    # flags on the landing.
    repeat = {"garden_lawn": (8, 8), "garden_paving": (2, 2), "garden_gravel": (10, 10)}
    for name, (rgba, emission, texture) in _MATERIALS.items():
        material = spec.add_material(
            name=name, rgba=list(rgba), emission=emission, specular=0.08, shininess=0.1,
            texrepeat=list(repeat.get(name, (4, 4))), texuniform=True,
        )
        if texture is not None:
            material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
    spec.worldbody.add_light(
        name="garden_sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0, 0, 10], dir=[0.35, 0.45, -0.82], castshadow=True,
        diffuse=[0.55, 0.52, 0.46], specular=[0.1, 0.1, 0.1], ambient=[0.0, 0.0, 0.0],
    )


def _garden_decor(
    body: mujoco.MjsBody, *, length: float, width: float, run_up: float,
    tread: float, num_steps: int, rise: float, margin: float, seed: int,
) -> list[TerrainGeometry]:
    """Non-colliding decoration for one patch, in the patch's frame.

    Everything that stands goes **beside** the patch, on the generator's border
    (y < 0 and y > width), so nothing a robot walks through is drawn in its
    lane. The moss sits on the riser faces, never on a tread, so no foot is drawn
    sinking into it. The landing gets only flat paving inlay at its far end.
    """
    out: list[TerrainGeometry] = []
    rng = np.random.default_rng(seed)  # its own: the generator's rng is untouched

    def add(gtype, size, pos, material):
        # Group 0, the terrain's own, and not the usual appearance group 2: the
        # viewer and the video hide a group whose geoms all collide when their
        # body also owns a non-colliding geom (`LiveViewer._collision_only_groups`).
        # Decoration in group 2 on the `terrain` body made exactly that true of
        # group 0, and the floor and every step vanished from the picture.
        geom = body.add_geom(
            type=gtype, size=list(size), pos=list(pos), material=material,
            contype=0, conaffinity=0, mass=0.0, group=0,
        )
        out.append(TerrainGeometry(geom=geom))

    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    sph = mujoco.mjtGeom.mjGEOM_SPHERE
    ell = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    top = num_steps * rise
    landing_x0 = run_up + num_steps * tread
    sides = (-1.0, 1.0)  # -1: beside y = 0, +1: beside y = width

    def beside(side: float, offset: float) -> float:
        return -offset if side < 0 else width + offset

    # Lawn over the border on both sides, 1 mm above its top so it wins the
    # depth test; the border's own top is at z = 0, like the floor's.
    if margin > 0.0:
        for side in sides:
            add(box, (length / 2.0, margin / 2.0, 0.0005),
                (length / 2.0, beside(side, margin / 2.0), 0.0005), "garden_lawn")

    # A gravel verge and a low stone cheek wall that follows the stair up.
    wall_t, wall_up = 0.05, 0.035
    for side in sides:
        add(box, (length / 2.0, 0.09, 0.0008),
            (length / 2.0, beside(side, wall_t + 0.09), 0.0012), "garden_gravel")
        for k in range(num_steps):
            h = (k + 1) * rise + wall_up
            add(box, (tread / 2.0, wall_t / 2.0, h / 2.0),
                (run_up + (k + 0.5) * tread, beside(side, wall_t / 2.0), h / 2.0),
                "garden_wall")
        h = top + wall_up
        add(box, ((length - landing_x0) / 2.0, wall_t / 2.0, h / 2.0),
            ((landing_x0 + length) / 2.0, beside(side, wall_t / 2.0), h / 2.0),
            "garden_wall")

    # Moss along the foot of every riser, in a few patches rather than one strip.
    for k in range(num_steps):
        x = run_up + k * tread - 0.004
        z0 = k * rise
        for y in rng.uniform(0.1, width - 0.1, size=3):
            half_w = rng.uniform(0.06, 0.16)
            h = min(rise * rng.uniform(0.3, 0.5), 0.02)
            if h < 0.002:
                continue
            add(box, (0.004, half_w, h / 2.0), (x, y, z0 + h / 2.0), "garden_moss")

    def lantern(x: float, y: float) -> None:
        add(box, (0.08, 0.08, 0.015), (x, y, 0.015), "garden_lantern")
        add(cyl, (0.032, 0.09), (x, y, 0.12), "garden_lantern")
        add(box, (0.075, 0.075, 0.014), (x, y, 0.224), "garden_lantern")
        add(box, (0.05, 0.05, 0.045), (x, y, 0.283), "garden_lantern_light")
        add(ell, (0.125, 0.125, 0.035), (x, y, 0.345), "garden_lantern")
        add(sph, (0.024,), (x, y, 0.392), "garden_lantern")

    def shrub(x: float, y: float, scale: float) -> None:
        add(ell, (0.16 * scale, 0.13 * scale, 0.10 * scale), (x, y, 0.06 * scale),
            "garden_shrub")
        add(ell, (0.10 * scale, 0.09 * scale, 0.08 * scale),
            (x + 0.05 * scale, y - 0.04 * scale, 0.11 * scale), "garden_shrub_light")
        for dx, dy in ((-0.06, 0.05), (0.07, 0.06), (0.0, -0.08)):
            add(sph, (0.018 * scale,), (x + dx * scale, y + dy * scale, 0.15 * scale),
                "garden_flower")

    def maple(x: float, y: float, scale: float) -> None:
        add(cyl, (0.035 * scale, 0.35 * scale), (x, y, 0.35 * scale), "garden_trunk")
        add(ell, (0.42 * scale, 0.38 * scale, 0.30 * scale), (x, y, 0.85 * scale),
            "garden_maple_red")
        add(ell, (0.28 * scale, 0.26 * scale, 0.22 * scale),
            (x + 0.18 * scale, y - 0.12 * scale, 1.05 * scale), "garden_maple_orange")
        add(ell, (0.24 * scale, 0.22 * scale, 0.18 * scale),
            (x - 0.20 * scale, y + 0.10 * scale, 1.00 * scale), "garden_maple_orange")

    if margin >= 0.6:
        for side in sides:
            lantern(run_up - 0.15, beside(side, 0.32))          # at the foot of the stair
            lantern(landing_x0 + 0.15, beside(side, 0.32))      # at its head
            shrub(run_up * 0.4, beside(side, 0.35), 1.0)
            shrub(landing_x0 + 0.9, beside(side, 0.40), 1.2)
            for x in rng.uniform(run_up, landing_x0, size=2):
                shrub(float(x), beside(side, rng.uniform(0.45, 0.8)), rng.uniform(0.6, 0.9))
    if margin >= 1.2:
        for side in sides:
            maple(landing_x0 + 2.2, beside(side, 0.9), 1.0)
            maple(length - 0.8, beside(side, 1.0), 0.8)
    return out
