"""
3D scene for the Volition renders, lit like a dark product studio: seamless floor,
a spotlight pool that follows the user, image-based lighting from a generated
softbox environment, and soft contact shadows.

Shadows come from a second, low-resolution "mask" render of the same geometry
(objects white, projected shadows black, floor grey, all unlit). The mask is
blurred and multiplied into the beauty pass, which gives soft shadows that VTK's
shadow maps don't.

Pose input is the per-frame state from sim/out/run.npz (see params.py for the
angle conventions). Rigid parts move via actor.user_matrix.
"""
import json
import math
import pathlib
import sys

import numpy as np
import pyvista as pv
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from params import USER  # noqa: E402

CAD = ROOT / "cad" / "out"
ENV_DIR = ROOT / "video" / "env"
pv.global_theme.allow_empty_mesh = True

L_TH, L_SH, H_ANK, HJC_Y = USER.thigh_len, USER.shank_len, USER.ankle_height, USER.hip_half_width
UPPER_ARM, FOREARM = 0.186 * USER.height * 0.92, 0.146 * USER.height
SKIN = (0.56, 0.565, 0.58)
LIGHT_DIR = np.array([0.30, 0.22, 1.0])          # towards the key light; sets the shadow direction


# ------------------------------------------------------------------ transforms
def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def tf(R=np.eye(3), t=(0, 0, 0)):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def shadow_matrix(height=0.002):
    lx, ly, lz = LIGHT_DIR
    S = np.eye(4)
    S[0, 2], S[1, 2], S[2, 2], S[2, 3] = -lx / lz, -ly / lz, 0.0, height
    return S


# --------------------------------------------------------------------- studio
def studio_cubemap(size=512, bright=False):
    """Dark room (or, with bright=True, a white sweep) with two large softboxes and a
    rim strip, as six cube faces."""
    env_dir = ENV_DIR / ("bright" if bright else "dark")
    env_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    u = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(u, u)

    def box(cx, cy, w, h, soft, inten):
        dx = np.clip((np.abs(X - cx) - w) / soft, 0, None)
        dy = np.clip((np.abs(Y - cy) - h) / soft, 0, None)
        return inten * np.exp(-(dx ** 2 + dy ** 2))

    base = (0.55 + 0.25 * (1 - (Y + 1) / 2)) if bright else (0.018 + 0.012 * (1 - (Y + 1) / 2))
    faces = {
        "posz": base + box(0, 0, 0.55, 0.35, 0.18, 1.6),                 # overhead softbox
        "posx": base + box(0.0, -0.35, 0.6, 0.18, 0.15, 0.9),            # front softbox
        "negx": base + box(0.0, -0.2, 0.08, 0.7, 0.06, 1.3),             # rim strip behind
        "posy": base + box(0.3, -0.2, 0.25, 0.5, 0.2, 0.45),
        "negy": base + box(-0.3, -0.2, 0.08, 0.6, 0.08, 0.8),
        "negz": np.full_like(X, 0.6 if bright else 0.012),
    }
    files = []
    for name in ("posx", "negx", "posy", "negy", "posz", "negz"):
        img = np.clip(faces[name], 0, 1) ** (1 / 2.2)
        rgb = np.stack([img * 0.98, img * 0.99, img], axis=-1)
        f = env_dir / f"{name}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(f)
        files.append(str(f))
    return pv.cubemap_from_filenames(files)


# --------------------------------------------------------------------- meshes
def capsule(p0, p1, r0, r1, prof=None, ox=None, sx=1.0, sy=1.0, fwd=(1, 0, 0), n_body=36, n_cap=10, n_circ=48):
    """Smooth tapered capsule from p0 to p1. prof(u) radius in body, ox(u) offset along fwd."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    ax = (p1 - p0) / L
    e1 = np.asarray(fwd, float) - ax * (np.asarray(fwd, float) @ ax)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0]) - ax * ax[1]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    rings = []
    for th in np.linspace(-math.pi / 2, 0, n_cap, endpoint=False)[1:]:
        rings.append((r0 * math.sin(th), r0 * math.cos(th), ox(0) if ox else 0.0))
    for u in np.linspace(0, 1, n_body):
        rings.append((u * L, prof(u) if prof else r0 + (r1 - r0) * u, ox(u) if ox else 0.0))
    for th in np.linspace(0, math.pi / 2, n_cap, endpoint=False)[1:]:
        rings.append((L + r1 * math.sin(th), r1 * math.cos(th), ox(1) if ox else 0.0))
    phi = np.linspace(0, 2 * math.pi, n_circ, endpoint=False)
    pts = [p0 + np.outer(np.full_like(phi, z), ax) + np.outer(sx * r * np.cos(phi) + o, e1) + np.outer(sy * r * np.sin(phi), e2)
           for z, r, o in rings]
    pts = np.vstack(pts + [p0 - ax * r0, p1 + ax * r1])
    nr, nc = len(rings), n_circ
    faces = []
    for i in range(nr - 1):
        for j in range(nc):
            a, b = i * nc + j, i * nc + (j + 1) % nc
            faces += [4, a, b, b + nc, a + nc]
    bot, top = nr * nc, nr * nc + 1
    for j in range(nc):
        faces += [3, bot, (j + 1) % nc, j, 3, top, (nr - 1) * nc + j, (nr - 1) * nc + (j + 1) % nc]
    return pv.PolyData(pts, np.array(faces)).compute_normals(auto_orient_normals=True, split_vertices=False)


def ellipsoid(center, radii, res=48):
    return pv.ParametricEllipsoid(*radii, u_res=res, v_res=res, w_res=res).translate(center, inplace=False).compute_normals()


def rounded_box(center, half, n=0.25, res=48):
    m = pv.ParametricSuperEllipsoid(xradius=half[0], yradius=half[1], zradius=half[2], n1=n, n2=n, u_res=res, v_res=res, w_res=res)
    return m.translate(center, inplace=False).compute_normals()


def head_points():
    """Approximate 10-20 electrode sites on the head ellipsoid (trunk frame)."""
    c = np.array([0.015, 0.0, 0.715])
    rad = np.array([0.104, 0.084, 0.118])
    sites = {"Cz": (0, 0), "C1": (0, 22), "C2": (0, -22), "C3": (0, 44), "C4": (0, -44), "C5": (0, 64), "C6": (0, -64),
             "FCz": (24, 0), "FC3": (24, 40), "FC4": (24, -40), "CPz": (-24, 0), "CP3": (-24, 40), "CP4": (-24, -40),
             "Pz": (-46, 0), "Fz": (46, 0), "F3": (44, 36), "F4": (44, -36), "P3": (-44, 36), "P4": (-44, -36),
             "T7": (0, 84), "T8": (0, -84)}
    out = {}
    for name, (fwd, lat) in sites.items():
        a, b = math.radians(fwd), math.radians(lat)
        out[name] = c + rad * np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
    return out


MOTOR_SITES = {"FEET": ["Cz", "C1", "C2", "FCz", "CPz"], "HANDS": ["C3", "C4", "FC3", "FC4", "CP3", "CP4", "C5", "C6"]}


def mannequin_parts():
    """Smooth, faceless studio mannequin in the exo's segment frames."""
    m = {}

    def torso_r(u):  # lateral half-width along the trunk
        return 0.170 - 0.034 * math.sin(math.pi * min(u / 0.72, 1.0)) ** 1.6 * (u < 0.72) - 0.02 * max(0.0, (u - 0.86) / 0.14) ** 2

    m["trunk"] = [capsule((-0.005, 0, -0.03), (-0.012, 0, 0.47), 0.15, 0.13, prof=torso_r, sx=0.70, sy=1.0, n_body=56, n_circ=64),
                  ellipsoid((-0.012, 0.168, 0.478), (0.058, 0.066, 0.056)),
                  ellipsoid((-0.012, -0.168, 0.478), (0.058, 0.066, 0.056)),
                  capsule((-0.008, 0, 0.49), (0.006, 0, 0.62), 0.050, 0.046),
                  ellipsoid((0.015, 0, 0.715), (0.098, 0.079, 0.114), res=64),
                  ellipsoid((0.058, 0, 0.660), (0.046, 0.054, 0.046))]
    for s in (1, -1):
        y, side = s * HJC_Y, "L" if s > 0 else "R"
        m[f"thigh_{side}"] = [capsule((0, y, 0.0), (0, y, -L_TH), 0.088, 0.054, prof=lambda u: 0.088 + (0.054 - 0.088) * u ** 1.2, sx=1.0, sy=0.95)]
        m[f"shank_{side}"] = [capsule((0, y, 0.0), (0, y, -(L_SH - 0.05)), 0.050, 0.031,
                                      prof=lambda u: 0.050 + (0.031 - 0.050) * u + 0.012 * math.exp(-((u - 0.28) / 0.15) ** 2),
                                      ox=lambda u: -0.010 * math.exp(-((u - 0.28) / 0.18) ** 2))]
    return m


def eeg_cap():
    head_c, rad = np.array([0.015, 0.0, 0.715]), np.array([0.104, 0.084, 0.118])
    cap = pv.ParametricEllipsoid(*rad, u_res=72, v_res=72, w_res=72).translate(head_c, inplace=False)
    cap = cap.clip(normal=(0.7, 0, -1), origin=(0.0, 0, 0.738), invert=False).compute_normals()   # keeps z > 0.738 + 0.7 x (hairline)
    elec = {k: pv.Sphere(radius=0.0095, center=p, theta_resolution=24, phi_resolution=24) for k, p in head_points().items()}
    pts = np.array([[-0.10, 0, 0.70], [-0.14, 0, 0.62], [-0.16, 0, 0.50], [-0.20, 0.02, 0.40], [-0.22, 0.03, 0.32]])
    cable = pv.Spline(pts, 40).tube(radius=0.0055, n_sides=20)
    amp = rounded_box((-0.225, 0.035, 0.307), (0.022, 0.035, 0.018), n=0.3)
    return cap, elec, cable, amp


def chair(x_front, seat_top):
    """Minimal studio chair: white shell seat and back on thin silver legs."""
    w, d, t = 0.46, 0.44, 0.05
    xc = x_front - d / 2
    shell = [rounded_box((xc, 0, seat_top - t / 2), (d / 2, w / 2, t / 2), n=0.22),
             rounded_box((x_front - d + 0.02, 0, seat_top + 0.27), (0.022, w / 2 - 0.01, 0.17), n=0.3)]
    legs = []
    for xx in (x_front - d + 0.04, x_front - 0.04):
        for yy in (-w / 2 + 0.04, w / 2 - 0.04):
            legs.append(capsule((xx, yy, 0.006), (xx, yy, seat_top - t), 0.011, 0.011, n_body=4, n_circ=24))
    return shell, legs


# --------------------------------------------------------------------- scene
class Scene:
    def __init__(self, window_size=(1920, 1080), chairs=(), mask_scale=0.5, ssaa=True, ibl=True):
        self.size = window_size
        self.p = pv.Plotter(off_screen=True, window_size=window_size, lighting="none")
        self.pm = pv.Plotter(off_screen=True, window_size=(int(window_size[0] * mask_scale), int(window_size[1] * mask_scale)), lighting="none")
        p, pm = self.p, self.pm
        p.set_background((0.0, 0.0, 0.0))
        pm.set_background((0.5, 0.5, 0.5))
        self.manifest = json.loads((CAD / "manifest.json").read_text())
        self.mats = self.manifest["materials"]
        self.S = shadow_matrix()
        self.rigid = {}          # segment -> list of (beauty actor, mask actor, shadow actor)
        self.indicator = {}
        self.electrodes = {}

        # a pool of light that travels with the user and falls off to the black backdrop
        pool = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=28, j_size=28, i_resolution=280, j_resolution=280)
        r = np.hypot(pool.points[:, 0] * 0.75, pool.points[:, 1])
        v = 0.16 * np.exp(-(r / 1.8) ** 2) + 0.045 * np.exp(-(r / 4.5) ** 2)
        pool["rgb"] = (np.clip(np.c_[v, v, v * 1.04], 0, 1) * 255).astype(np.uint8)
        self.pool = p.add_mesh(pool, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)
        pm.add_mesh(pv.Plane(center=(8, 0, 0), direction=(0, 0, 1), i_size=200, j_size=200), color=(0.5, 0.5, 0.5), lighting=False)

        for seg, info in self.manifest["segments"].items():
            self.rigid[seg] = []
            for mat, fname in info["files"].items():
                mesh = pv.read(CAD / "parts" / fname).compute_normals(split_vertices=True, feature_angle=35)
                trio = self._add(mesh, mat)
                if mat == "indicator":
                    if seg.startswith("thigh"):
                        self.indicator["hip_" + seg[-1]] = trio[0]
                    elif seg.startswith("shank"):
                        self.indicator["knee_" + seg[-1]] = trio[0]
                self.rigid[seg].append(trio)
        for seg, meshes in mannequin_parts().items():
            key = "pelvis" if seg == "trunk" else seg
            for mesh in meshes:
                self.rigid[key].append(self._add(mesh, rgb=SKIN, rough=0.72, metal=0.0))
        cap, elec, cable, amp = eeg_cap()
        self.rigid["pelvis"].append(self._add(cap, "pad"))
        self.rigid["pelvis"].append(self._add(cable, "rubber"))
        self.rigid["pelvis"].append(self._add(amp, "shell"))
        for name, mesh in elec.items():
            trio = self._add(mesh, rgb=(0.80, 0.81, 0.83), rough=0.25, metal=0.9, shadow=False)
            self.electrodes[name] = trio[0]
            self.rigid["pelvis"].append(trio)

        # arms and crutches are rebuilt every frame (their shape depends on IK)
        self.dyn = {}
        for s in ("L", "R"):
            for part, mat in (("arm", None), ("alu", "alu"), ("grip", "rubber"), ("cuff", "carbon")):
                mesh = pv.PolyData()
                if mat:
                    mm = self.mats[mat]
                    p.add_mesh(mesh, color=mm["rgb"], pbr=True, metallic=mm["metallic"], roughness=mm["roughness"], smooth_shading=True)
                else:
                    p.add_mesh(mesh, color=SKIN, pbr=True, metallic=0.0, roughness=0.72, smooth_shading=True)
                pm.add_mesh(mesh, color="white", lighting=False)
                self.dyn[(s, part)] = mesh
        self.dyn_shadow = pv.PolyData()
        pm.add_mesh(self.dyn_shadow, color="black", lighting=False)

        self.chairs = []
        for xf, top in chairs:
            shell, legs = chair(xf, top)
            trios = [self._add(m, rgb=(0.93, 0.93, 0.94), rough=0.35, metal=0.0) for m in shell]
            trios += [self._add(m, rgb=(0.78, 0.79, 0.81), rough=0.25, metal=1.0) for m in legs]
            self.chairs.append(trios)
            self.set_chair(len(self.chairs) - 1, (0, 0, 0))

        if ibl:
            try:
                p.set_environment_texture(studio_cubemap(), is_srgb=True)
            except Exception as exc:  # older VTK without IBL: fall back to lights only
                print("IBL unavailable:", exc)
        self.key = pv.Light(position=(1.6, 1.2, 5.0), focal_point=(0, 0, 0.6), color=(1.0, 0.97, 0.93), intensity=1.6,
                            positional=True, cone_angle=26, exponent=6, light_type="scene light")
        self.rim = pv.Light(position=(-4.0, -2.5, 2.6), focal_point=(0, 0, 0.9), color=(0.80, 0.88, 1.0), intensity=0.9, light_type="scene light")
        self.fill = pv.Light(position=(3.0, -3.5, 1.4), focal_point=(0, 0, 0.8), color=(1.0, 1.0, 1.0), intensity=0.25, light_type="scene light")
        for light in (self.key, self.rim, self.fill):
            p.add_light(light)
        if ssaa:
            p.enable_anti_aliasing("ssaa")

    def _add(self, mesh, mat=None, rgb=None, rough=None, metal=None, shadow=True):
        if mat:
            mm = self.mats[mat]
            rgb, rough, metal = mm["rgb"], mm["roughness"], mm["metallic"]
        a = self.p.add_mesh(mesh, color=rgb, pbr=True, metallic=metal, roughness=rough, smooth_shading=True)
        if mat == "led":
            a.prop.ambient = 0.9
        am = self.pm.add_mesh(mesh, color="white", lighting=False)
        sh = self.pm.add_mesh(mesh, color="black", lighting=False) if shadow else None
        return (a, am, sh)

    def set_chair(self, i, offset, visible=True):
        M = tf(t=offset)
        for a, am, sh in self.chairs[i]:
            a.user_matrix, am.user_matrix, sh.user_matrix = M, M, self.S @ M
            a.visibility = am.visibility = sh.visibility = visible

    # ------------------------------------------------------------- per frame
    def pose(self, st):
        """st: pelvis_xz, trunk_pitch, phi_thigh_L/R, phi_shank_L/R, psi_foot_L/R, hand_xyz_L/R,
        crutch_tip_xyz_L/R, util (joint -> 0..1), sat (joint -> bool), mi (FEET/HANDS -> 0..1)."""
        hx, hz = st["pelvis_xz"]
        H = np.array([hx, 0.0, hz])
        M = {"pelvis": tf(ry(st["trunk_pitch"]), H)}
        for s in ("L", "R"):
            ft, fs, pf = st[f"phi_thigh_{s}"], st[f"phi_shank_{s}"], st[f"psi_foot_{s}"]
            K = H + np.array([L_TH * math.sin(ft), 0, -L_TH * math.cos(ft)])
            A = K + np.array([L_SH * math.sin(fs), 0, -L_SH * math.cos(fs)])
            M[f"thigh_{s}"], M[f"shank_{s}"], M[f"foot_{s}"] = tf(ry(-ft), H), tf(ry(-fs), K), tf(ry(-pf), A)
        for seg, lst in self.rigid.items():
            for a, am, sh in lst:
                a.user_matrix = M[seg]
                am.user_matrix = M[seg]
                if sh is not None:
                    sh.user_matrix = self.S @ M[seg]
        self._arms_and_crutches(M["pelvis"], st)
        for j, a in self.indicator.items():
            u = float(np.clip(st.get("util", {}).get(j, 0.0), 0, 1))
            if st.get("sat", {}).get(j, False):
                rgb = (1.0, 0.23, 0.19)
            else:
                rgb = tuple(np.array([0.95, 0.97, 1.0]) * (1 - u) + np.array([1.0, 0.58, 0.0]) * u)
            a.prop.color = rgb
            a.prop.ambient = 0.85
        mi = st.get("mi", {})
        for name, a in self.electrodes.items():
            level = max([mi.get(c, 0.0) for c, sites in MOTOR_SITES.items() if name in sites] + [0.0])
            a.prop.color = tuple(np.array([0.80, 0.81, 0.83]) * (1 - level) + np.array([0.39, 0.82, 1.0]) * level)
            a.prop.ambient = 0.1 + 0.85 * level
        self.pool.user_matrix = tf(t=(hx + 0.25, 0.0, 0.0))
        self.key.position = (hx + 1.6, 1.2, 5.0)
        self.key.focal_point = (hx + 0.1, 0.0, 0.6)
        return M

    def _arms_and_crutches(self, Mp, st):
        shadow_parts = []
        for s, sign in (("L", 1), ("R", -1)):
            sh = (Mp @ np.array([-0.012, sign * 0.175, 0.478, 1.0]))[:3]
            hand = np.asarray(st[f"hand_xyz_{s}"], float)
            tip = np.asarray(st[f"crutch_tip_xyz_{s}"], float)
            d = hand - sh
            dist = np.linalg.norm(d)
            reach = UPPER_ARM + FOREARM - 1e-3
            if dist > reach:
                hand = sh + d / dist * reach
                d, dist = hand - sh, reach
            u = d / dist
            pole = np.array([-0.6, sign * 1.0, -0.15])
            pole -= u * (pole @ u)
            pole /= np.linalg.norm(pole)
            a = (UPPER_ARM ** 2 - FOREARM ** 2 + dist ** 2) / (2 * dist)
            elbow = sh + u * a + pole * math.sqrt(max(UPPER_ARM ** 2 - a ** 2, 0.0))
            arm = (capsule(sh, elbow, 0.047, 0.038, n_body=16, n_circ=36) + capsule(elbow, hand, 0.037, 0.028, n_body=16, n_circ=36)
                   + ellipsoid(hand + np.array([0.0, 0.0, 0.004]), (0.050, 0.031, 0.042), res=28))
            fa = (elbow - hand) / np.linalg.norm(elbow - hand)
            grip_dir = np.array([1.0, 0.0, 0.0]) - fa * fa[0]
            grip_dir /= np.linalg.norm(grip_dir)
            handle = hand - grip_dir * 0.045
            cuff = hand + fa * 0.17
            alu = (capsule(tip + np.array([0, 0, 0.045]), handle, 0.0105, 0.0105, n_body=4, n_circ=28)
                   + capsule(handle, cuff, 0.009, 0.009, n_body=4, n_circ=28))
            grip = (capsule(tip + np.array([0, 0, 0.012]), tip + np.array([0, 0, 0.05]), 0.016, 0.014, n_body=4, n_circ=28)
                    + capsule(hand - grip_dir * 0.05, hand + grip_dir * 0.07, 0.015, 0.015, n_body=6, n_circ=28))
            cuff_m = pv.Cylinder(center=cuff, direction=fa, radius=0.044, height=0.032, resolution=48, capping=False).triangulate().compute_normals()
            self.dyn[(s, "arm")].copy_from(arm)
            self.dyn[(s, "alu")].copy_from(alu)
            self.dyn[(s, "grip")].copy_from(grip)
            self.dyn[(s, "cuff")].copy_from(cuff_m)
            shadow_parts += [arm, alu, grip]
        sh = shadow_parts[0]
        for m in shadow_parts[1:]:
            sh = sh + m
        self.dyn_shadow.copy_from(sh.transform(self.S, inplace=False))

    def camera(self, pos, focal, up=(0, 0, 1), view_angle=26):
        for pl in (self.p, self.pm):
            pl.camera.position = tuple(pos)
            pl.camera.focal_point = tuple(focal)
            pl.camera.up = tuple(up)
            pl.camera.view_angle = view_angle
            pl.camera.clipping_range = (0.05, 120)

    def image(self):
        """Beauty pass with soft shadows multiplied in; float32 RGB 0..255."""
        self.p.render()
        img = self.p.screenshot(return_img=True).astype(np.float32)
        self.pm.render()
        m = self.pm.screenshot(return_img=True)[..., 0].astype(np.float32) / 255.0
        shadow = (m < 0.25).astype(np.float32)
        obj = (m > 0.75).astype(np.float32)
        soft = 0.55 * ndimage.gaussian_filter(shadow, 2.5) + 0.45 * ndimage.gaussian_filter(shadow, 10.0)
        k = np.clip(soft * (1.0 - ndimage.gaussian_filter(obj, 0.8)), 0, 1)
        k = ndimage.zoom(k, (img.shape[0] / k.shape[0], img.shape[1] / k.shape[1]), order=1)[: img.shape[0], : img.shape[1]]
        img *= (1.0 - 0.78 * k)[..., None]
        return img
