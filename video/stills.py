"""
Product renders for the documentation page.

  white studio : exo_iso / exo_front / exo_side, actuator_cutaway / actuator_exploded
  dark studio  : hero.jpg, a frame of the simulated walk without the overlay

    python video/stills.py
"""
import json
import math
import pathlib
import sys

import numpy as np
import pyvista as pv
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "video"))
from scene import Scene, studio_cubemap  # noqa: E402

CAD = ROOT / "cad" / "out"
OUT = ROOT / "video" / "out" / "stills"
MAN = json.loads((CAD / "manifest.json").read_text())


def white_studio(size):
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.set_background("white")
    p.set_environment_texture(studio_cubemap(bright=True), is_srgb=True)
    for pos, inten in [((3, 2, 5), 0.9), ((-4, -2, 2), 0.35), ((-2, 4, 1.5), 0.45)]:
        p.add_light(pv.Light(position=pos, focal_point=(0, 0, 0.6), intensity=inten, light_type="scene light"))
    p.enable_anti_aliasing("ssaa")
    return p


def add(p, mesh, mat):
    mm = MAN["materials"][mat]
    p.add_mesh(mesh, color=mm["rgb"], pbr=True, metallic=mm["metallic"], roughness=mm["roughness"],
               smooth_shading=True, split_sharp_edges=True)


def contact_shadow(p, center, rx, ry, depth=0.16):
    """Unlit disc whose colour fades from soft grey to the white background."""
    disc = pv.Disc(center=center, inner=0, outer=1.0, normal=(0, 0, 1), r_res=40, c_res=96)
    r = np.hypot(disc.points[:, 0] - center[0], disc.points[:, 1] - center[1])
    disc.points[:, 0] = center[0] + (disc.points[:, 0] - center[0]) * rx
    disc.points[:, 1] = center[1] + (disc.points[:, 1] - center[1]) * ry
    v = 1.0 - depth * np.exp(-(r / 0.45) ** 2) - 0.04 * np.exp(-(r / 0.9) ** 2)
    disc["rgb"] = (np.c_[v, v, v] * 255).astype(np.uint8)
    p.add_mesh(disc, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)


def exo(name, view, size=(1200, 1500)):
    p = white_studio(size)
    orig = MAN["geometry"]["segment_origins_standing"]
    for seg, info in MAN["segments"].items():
        for mat, fname in info["files"].items():
            add(p, pv.read(CAD / "parts" / fname).translate(orig[seg], inplace=False), mat)
    contact_shadow(p, (0.04, 0.0, 0.0005), 0.55, 0.62)
    focal = np.array([0.0, 0.0, 0.70])
    p.camera.focal_point = tuple(focal)
    p.camera.position = tuple(focal + np.array(view))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 26
    p.screenshot(OUT / f"{name}.png")


def actuator(view, direction, size, zoom):
    p = white_studio(size)
    for f in sorted((CAD / "actuator").glob(f"{view}__*.stl")):
        add(p, pv.read(f), f.stem.split("__")[1])
    p.view_vector(direction, viewup=(0, 0, 1))
    p.reset_camera()
    p.camera.zoom(zoom)
    p.screenshot(OUT / f"actuator_{view}.png")


def hero(t=21.4, size=(2400, 1100)):
    """A frame of the simulated walk, dark studio, no overlay."""
    from render import Run, pose_state
    run = Run()
    scene = Scene(window_size=size)
    scene.pose(pose_state(run, t, mi={"FEET": 0.35, "HANDS": 0.0}))
    hx, _ = run.at("pelvis_xz", t)
    focal = np.array([hx + 0.08, 0.0, 0.90])
    a, e, d = math.radians(56), math.radians(5), 4.3
    pos = focal + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    scene.camera(pos, focal, view_angle=26)
    img = np.clip(scene.image(), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(OUT / "hero.jpg", quality=92)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    exo("exo_iso", (2.6, 1.9, 0.75))
    exo("exo_front", (3.3, 0.0, 0.25))
    exo("exo_side", (0.0, 3.3, 0.25))
    actuator("cutaway", (0.85, 1.0, 0.65), (1600, 1100), 1.25)
    actuator("exploded", (1.0, 0.42, 0.42), (1800, 900), 1.45)
    hero()
    print("stills ->", OUT)
