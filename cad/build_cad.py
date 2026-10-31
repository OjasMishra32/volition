"""
Parametric CAD for the Volition exoskeleton (CadQuery / OpenCascade B-rep).

    python cad/build_cad.py

Writes cad/out/:
  parts/<segment>__<material>.stl  segment-local frames (see params.py), for rendering
  parts/<segment>.step             one STEP per segment, left side (right = mirror in Y)
  Volition_standing.step           full coloured assembly, standing pose
  EXA-100_actuator.step            actuator with internals
  actuator/<view>__<material>.stl  actuator cut-away and exploded views
  volition.glb                     web-viewer model, one node per segment/material
  manifest.json                    joint positions, materials, part masses (volume x density)
"""
import json
import math
import pathlib
import sys

import cadquery as cq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from params import ACT, STRUCT, USER  # noqa: E402

OUT = ROOT / "cad" / "out"
V = cq.Vector

L_TH = USER.thigh_len
L_SH = USER.shank_len
H_ANK = USER.ankle_height
HJC_Y = USER.hip_half_width
R_ACT = ACT.housing_od / 2
W_ACT = ACT.housing_len

# name: (rgb, metallic, roughness, density kg/m^3 for mass estimates)
MATERIALS = {
    "carbon":  ((0.115, 0.118, 0.125), 0.55, 0.38, 1550),   # space-black anodised / carbon
    "alu":     ((0.780, 0.790, 0.810), 1.00, 0.24, 2700),   # bead-blasted silver
    "accent":  ((1.000, 0.420, 0.040), 0.10, 0.35, 1200),   # signal orange
    "shell":   ((0.945, 0.945, 0.950), 0.00, 0.32, 1150),   # gloss white polymer
    "pad":     ((0.200, 0.205, 0.215), 0.00, 0.90, 200),    # graphite knit
    "rubber":  ((0.070, 0.070, 0.075), 0.00, 0.80, 1100),
    "fabric":  ((0.300, 0.305, 0.320), 0.00, 0.85, 400),
    "led":     ((0.250, 0.780, 1.000), 0.00, 0.20, 1200),
    "estop":   ((0.900, 0.120, 0.100), 0.00, 0.35, 1200),
    "yellow":  ((0.960, 0.820, 0.200), 0.00, 0.45, 1200),
    "steel":   ((0.560, 0.575, 0.600), 1.00, 0.30, 7850),
    "copper":  ((0.850, 0.520, 0.300), 1.00, 0.30, 8900),
    "magnet":  ((0.300, 0.310, 0.335), 0.80, 0.40, 7500),
    "pcb":     ((0.090, 0.330, 0.200), 0.00, 0.50, 1850),
    "indicator": ((1.000, 0.420, 0.040), 0.00, 0.30, 1200),  # actuator status ring (lit per joint in renders)
}


class Part:
    """A rigid segment: shapes grouped by material, in the segment-local frame."""

    def __init__(self, name):
        self.name = name
        self.shapes = {}

    def add(self, mat, *shapes):
        assert mat in MATERIALS, mat
        self.shapes.setdefault(mat, []).extend(shapes)
        return self

    def merge(self, other):
        for mat, shapes in other.shapes.items():
            self.add(mat, *shapes)
        return self

    def compounds(self):
        return {m: cq.Compound.makeCompound(s) for m, s in self.shapes.items()}

    def mirrored(self, name):
        out = Part(name)
        for mat, shapes in self.shapes.items():
            out.add(mat, *[s.mirror("XZ") for s in shapes])
        return out


# ------------------------------------------------------------------ primitives
def cyl(r, p0, p1, angle=360):
    p0, p1 = V(*p0), V(*p1)
    d = p1 - p0
    return cq.Solid.makeCylinder(r, d.Length, p0, d.normalized(), angle)


def ring(ro, ri, p0, p1):
    return cyl(ro, p0, p1).cut(cyl(ri, p0, p1))


def box(size, center, fillet=0.0, sel=None):
    wp = cq.Workplane().box(*size)
    fillet = min(fillet, 0.45 * min(size))  # OCC fails when the radius reaches half a side
    if fillet:
        wp = (wp.edges(sel) if sel else wp.edges()).fillet(fillet)
    return wp.val().translate(V(*center))


def bar(p0, p1, w, d, wall=0.0, fillet=0.0):
    """Rectangular tube (w along X, d along Y) on the line p0 -> p1."""
    p0, p1 = V(*p0), V(*p1)
    length = (p1 - p0).Length
    wp = cq.Workplane().box(w, d, length)
    fillet = min(fillet, 0.45 * min(w, d))
    if wall:
        fillet = min(fillet, 0.9 * wall + 0.5 * min(w - 2 * wall, d - 2 * wall) * 0.5)
    if fillet:
        wp = wp.edges("|Z").fillet(fillet)
    s = wp.val()
    if wall:
        s = s.cut(cq.Workplane().box(w - 2 * wall, d - 2 * wall, length + 0.01).val())
    z, dv = V(0, 0, 1), (p1 - p0).normalized()
    axis = z.cross(dv)
    if axis.Length > 1e-9:
        s = s.rotate(V(0, 0, 0), axis, math.degrees(math.acos(max(-1.0, min(1.0, z.dot(dv))))))
    return s.translate((p0 + p1) * 0.5)


def arc_shell(r_out, r_in, z0, z1, cx, cy, a0, a1):
    """Partial cylindrical shell about a vertical axis through (cx, cy); angles from +X."""
    outer = cq.Solid.makeCylinder(r_out, z1 - z0, V(0, 0, 0), V(0, 0, 1), a1 - a0)
    s = outer.cut(cq.Solid.makeCylinder(r_in, z1 - z0, V(0, 0, 0), V(0, 0, 1)))
    return s.rotate(V(0, 0, 0), V(0, 0, 1), a0).translate(V(cx, cy, z0))


def ellipse_ring(ax_o, by_o, ax_i, by_i, z0, h, keep=None):
    s = (cq.Workplane("XY").ellipse(ax_o, by_o).extrude(h)
         .cut(cq.Workplane("XY").ellipse(ax_i, by_i).extrude(h)).val())
    if keep == "back":
        s = s.cut(box((1, 1, 1), (0.5, 0, h / 2)))
    elif keep == "front":
        s = s.cut(box((1, 1, 1), (-0.5, 0, h / 2)))
    return s.translate(V(0, 0, z0))


def scaled(shape, sx, sy, sz):
    return shape.transformGeometry(cq.Matrix([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0]]))


def line_y(p0, p1, z):
    """Y of the straight line p0=(y0,z0) -> p1=(y1,z1) at height z."""
    (y0, z0), (y1, z1) = p0, p1
    return y0 + (y1 - y0) * (z - z0) / (z1 - z0)


# ---------------------------------------------------------- EXA-100 (simplified)
def exa_housing(yc, z=0.0):
    """Stator side of the actuator (bolts to the proximal segment). Axis along Y,
    output flange on the lateral (+Y) face."""
    y0, y1 = yc - W_ACT / 2, yc + W_ACT / 2
    p = Part("exa_housing")
    body = cyl(R_ACT, (0, y0 + 0.004, z), (0, y1 - 0.006, z))
    for k in range(4):  # cooling grooves
        yg = y0 + 0.012 + k * 0.008
        body = body.cut(ring(R_ACT + 0.001, R_ACT - 0.0022, (0, yg, z), (0, yg + 0.0025, z)))
    p.add("alu", body)
    p.add("carbon", cyl(R_ACT - 0.003, (0, y0, z), (0, y0 + 0.004, z)))
    p.add("accent", ring(R_ACT + 0.0008, R_ACT - 0.002, (0, y1 - 0.013, z), (0, y1 - 0.007, z)))
    # power/CAN-FD connector boss, pointing rearward
    p.add("carbon", cyl(0.0075, (-R_ACT + 0.004, yc - 0.008, z), (-R_ACT - 0.012, yc - 0.008, z)))
    p.add("accent", cyl(0.0050, (-R_ACT - 0.012, yc - 0.008, z), (-R_ACT - 0.016, yc - 0.008, z)))
    return p


def exa_flange(yc, z=0.0):
    """Rotor/output side of the actuator (moves with the distal segment)."""
    y1 = yc + W_ACT / 2
    p = Part("exa_flange")
    p.add("alu", cyl(0.045, (0, y1 - 0.006, z), (0, y1 + 0.003, z)))
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        x, zz = 0.036 * math.cos(a), z + 0.036 * math.sin(a)
        p.add("steel", cyl(0.0035, (x, y1 + 0.003, zz), (x, y1 + 0.0055, zz)))
    p.add("indicator", ring(0.016, 0.010, (0, y1 + 0.003, z), (0, y1 + 0.0050, z)))
    p.add("carbon", cyl(0.010, (0, y1 + 0.003, z), (0, y1 + 0.0055, z)))
    return p


# -------------------------------------------------------------------- segments
def build_pelvis():
    """Pelvis frame + backpack + hip actuator housings. Origin: hip-axis midpoint."""
    p = Part("pelvis")
    left = Part("pelvis_left")
    hy = STRUCT.hip_act_y
    left.merge(exa_housing(hy))
    # abduction hinge boss on top of the hip housing (passive ab/ad DOF, spring-centred)
    left.add("carbon", box((0.056, 0.050, 0.030), (0, hy - 0.006, R_ACT + 0.010), 0.006))
    left.add("alu", cyl(0.0085, (-0.036, hy - 0.006, R_ACT + 0.030), (0.036, hy - 0.006, R_ACT + 0.030)))
    left.add("accent", cyl(0.0105, (0.028, hy - 0.006, R_ACT + 0.030), (0.036, hy - 0.006, R_ACT + 0.030)))
    p.merge(left).merge(left.mirrored("pelvis_right"))

    # posterior pelvic band (carbon) and anterior belt (padded strap)
    p.add("carbon", ellipse_ring(0.152, 0.212, 0.140, 0.200, 0.052, 0.046, keep="back"))
    p.add("pad", ellipse_ring(0.136, 0.204, 0.132, 0.200, 0.058, 0.034, keep="front"))
    p.add("alu", box((0.012, 0.050, 0.030), (0.137, 0, 0.075), 0.004))

    # lumbar plate + backpack (battery, electronics), E-stop, status LEDs
    bx, by, bz = STRUCT.backpack_center
    dx, dy, dz = STRUCT.backpack_size
    p.add("carbon", box((0.012, 0.17, 0.24), (-0.150, 0, 0.14), 0.004))
    pack = box((dx, dy, dz), (bx, by, bz), 0.018)
    for k in range(5):  # side vents
        for s in (1, -1):
            pack = pack.cut(box((0.050, 0.010, 0.004), (bx, s * dy / 2, bz - 0.05 + 0.022 * k)))
    p.add("shell", pack)
    p.add("carbon", box((0.022, 0.24, 0.20), (bx - dx / 2 - 0.008, 0, bz - 0.02), 0.007))  # battery
    p.add("accent", box((0.010, 0.06, 0.012), (bx - dx / 2 - 0.020, 0, bz + 0.072), 0.003))  # latch
    p.add("led", box((0.006, 0.22, 0.006), (bx - dx / 2 - 0.0005, 0, bz + dz / 2 - 0.022)))
    p.add("carbon", bar((bx - 0.02, -0.07, bz + dz / 2 + 0.002), (bx - 0.02, 0.07, bz + dz / 2 + 0.002), 0.016, 0.016, fillet=0.006))
    ex, ey, ez = bx + 0.01, -0.105, bz + dz / 2
    p.add("yellow", cyl(0.022, (ex, ey, ez - 0.002), (ex, ey, ez + 0.008)))
    p.add("estop", cyl(0.016, (ex, ey, ez + 0.008), (ex, ey, ez + 0.024)))

    # trunk uprights + chest strap (for users with high-level lesions / poor trunk control)
    for s in (1, -1):
        p.add("carbon", bar((-0.145, s * 0.10, bz + dz / 2 - 0.03), (-0.135, s * 0.10, 0.47), 0.012, 0.030, fillet=0.004))
    p.add("pad", ellipse_ring(0.138, 0.178, 0.130, 0.170, 0.415, 0.045))
    p.add("alu", box((0.012, 0.045, 0.030), (0.137, 0, 0.437), 0.004))
    return p


def build_thigh():
    """Thigh link (left). Origin on the hip axis; knee axis at z = -L_TH."""
    p = Part("thigh_L")
    hy, ky = STRUCT.hip_act_y, STRUCT.knee_act_y
    yf = hy + W_ACT / 2 + 0.003          # lateral face of hip output flange
    p.merge(exa_flange(hy))
    # output arm off the hip flange
    p.add("carbon", cyl(0.032, (0, yf, 0), (0, yf + 0.010, 0)))
    p.add("carbon", bar((0, yf + 0.005, 0), (0, yf + 0.005, -0.11), 0.048, 0.010, fillet=0.008))
    # telescoping link: outer tube (upper), inner tube (lower), clamp collar
    top, bot = (yf + 0.002, -0.09), (ky + 0.014, -L_TH + 0.065)
    z_clamp = -0.26
    y_clamp = line_y(top, bot, z_clamp)
    p.add("carbon", bar((0, top[0], top[1]), (0, y_clamp, z_clamp), 0.036, 0.020, wall=0.0025, fillet=0.006))
    p.add("alu", bar((0, y_clamp, z_clamp), (0, bot[0], bot[1]), 0.029, 0.015, wall=0.002, fillet=0.004))
    p.add("alu", bar((0, line_y(top, bot, z_clamp + 0.02), z_clamp + 0.02), (0, line_y(top, bot, z_clamp - 0.02), z_clamp - 0.02), 0.044, 0.028, fillet=0.006))
    for dz in (0.011, -0.011):
        p.add("steel", cyl(0.0045, (0.022, y_clamp, z_clamp + dz), (0.030, y_clamp, z_clamp + dz)))
    for k in range(6):  # length scale marks on the inner tube
        zz = z_clamp - 0.035 - 0.012 * k
        p.add("carbon", bar((0, line_y(top, bot, zz), zz), (0, line_y(top, bot, zz - 0.0015), zz - 0.0015), 0.0295, 0.0155))
    # cable conduit along the link (CAN-FD + 48 V to the knee drive)
    p.add("rubber", bar((-0.022, top[0], top[1] - 0.02), (-0.020, bot[0], bot[1] + 0.02), 0.008, 0.008, fillet=0.0035))
    # knee actuator housing + socket gripping it
    p.merge(exa_housing(ky, z=-L_TH))
    p.add("carbon", box((0.054, W_ACT + 0.004, 0.034), (0, ky, -L_TH + R_ACT + 0.006), 0.006))
    # posterior thigh cuff: shell + liner + anterior strap + bracket to the link
    zc = -STRUCT.thigh_cuff_at * L_TH
    p.add("shell", arc_shell(0.080, 0.074, zc - 0.045, zc + 0.045, 0, HJC_Y, 95, 235))
    p.add("pad", arc_shell(0.074, 0.069, zc - 0.042, zc + 0.042, 0, HJC_Y, 97, 233))
    p.add("pad", arc_shell(0.077, 0.074, zc - 0.022, zc + 0.022, 0, HJC_Y, 235, 455))
    yl = line_y(top, bot, zc)
    p.add("carbon", box((0.030, yl - (HJC_Y + 0.077), 0.060), (-0.006, (yl + HJC_Y + 0.077) / 2, zc), 0.004))
    p.add("accent", box((0.004, 0.014, 0.030), (0.077, HJC_Y, zc), 0.0015))  # strap buckle
    return p


def build_shank():
    """Shank link (left). Origin on the knee axis; ankle axis at z = -L_SH."""
    p = Part("shank_L")
    ky, ay = STRUCT.knee_act_y, STRUCT.ankle_hinge_y
    yf = ky + W_ACT / 2 + 0.003
    p.merge(exa_flange(ky))
    p.add("carbon", cyl(0.030, (0, yf, 0), (0, yf + 0.009, 0)))
    p.add("carbon", bar((0, yf + 0.0045, 0), (0, yf + 0.0045, -0.10), 0.044, 0.009, fillet=0.008))
    top, bot = (yf + 0.002, -0.085), (ay + 0.012, -L_SH + 0.045)
    z_clamp = -0.22
    y_clamp = line_y(top, bot, z_clamp)
    p.add("carbon", bar((0, top[0], top[1]), (0, y_clamp, z_clamp), 0.032, 0.018, wall=0.0025, fillet=0.005))
    p.add("alu", bar((0, y_clamp, z_clamp), (0, bot[0], bot[1]), 0.026, 0.013, wall=0.002, fillet=0.004))
    p.add("alu", bar((0, line_y(top, bot, z_clamp + 0.018), z_clamp + 0.018), (0, line_y(top, bot, z_clamp - 0.018), z_clamp - 0.018), 0.040, 0.026, fillet=0.006))
    for dz in (0.010, -0.010):
        p.add("steel", cyl(0.004, (0.020, y_clamp, z_clamp + dz), (0.027, y_clamp, z_clamp + dz)))
    # ankle hinge (outer knuckle) + dorsiflexion-assist spring cartridge (AFO)
    p.add("alu", bar((0, bot[0], bot[1]), (0, ay + 0.012, -L_SH + 0.004), 0.030, 0.012, fillet=0.005))
    p.add("alu", cyl(0.022, (0, ay + 0.004, -L_SH), (0, ay + 0.018, -L_SH)))
    p.add("accent", cyl(0.012, (0, ay + 0.018, -L_SH), (0, ay + 0.021, -L_SH)))
    p.add("carbon", cyl(0.011, (-0.030, ay + 0.012, -L_SH + 0.03), (-0.030, ay + 0.012, -L_SH + 0.115)))
    p.add("accent", cyl(0.0125, (-0.030, ay + 0.012, -L_SH + 0.075), (-0.030, ay + 0.012, -L_SH + 0.100)))
    p.add("carbon", box((0.030, 0.012, 0.016), (-0.016, ay + 0.012, -L_SH + 0.105), 0.004))
    # anterior tibial cuff + posterior calf strap
    zc = -STRUCT.shank_cuff_at * L_SH
    cx = -0.008
    p.add("shell", arc_shell(0.071, 0.065, zc - 0.040, zc + 0.040, cx, HJC_Y, -55, 85))
    p.add("pad", arc_shell(0.065, 0.060, zc - 0.037, zc + 0.037, cx, HJC_Y, -53, 83))
    p.add("pad", arc_shell(0.068, 0.065, zc - 0.020, zc + 0.020, cx, HJC_Y, 85, 305))
    yl = line_y(top, bot, zc)
    p.add("carbon", box((0.028, yl - (HJC_Y + 0.064), 0.055), (0.004, (yl + HJC_Y + 0.064) / 2, zc), 0.004))
    return p


def foot_outline(n=72):
    """Plan-view outline of a shoe, ankle centre at the origin."""
    heel, toe = -0.068, USER.foot_len - 0.068
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s = math.cos(t), math.sin(t)
        x = (toe + heel) / 2 + (toe - heel) / 2 * math.copysign(abs(c) ** 0.8, c)
        u = (x - heel) / (toe - heel)
        half_w = 0.034 + 0.020 * math.sin(math.pi * min(1.0, u * 1.15)) ** 0.7
        y = HJC_Y + 0.006 * (u - 0.5) + half_w * math.copysign(abs(s) ** 0.7, s)
        pts.append((x, y))
    return pts


def build_foot():
    """Shoe + AFO footplate + stirrup (left). Origin on the ankle axis."""
    p = Part("foot_L")
    ay = STRUCT.ankle_hinge_y
    outline = foot_outline()
    sole = lambda h: cq.Workplane("XY").polyline(outline).close().extrude(h).val()  # noqa: E731
    p.add("rubber", sole(0.012).translate(V(0, 0, -H_ANK)))
    p.add("carbon", sole(0.005).translate(V(0, 0, -H_ANK + 0.012)))
    upper = scaled(cq.Solid.makeSphere(1.0, angleDegrees1=0, angleDegrees2=90), 0.142, 0.050, 0.060)
    upper = upper.translate(V(0.068, HJC_Y, -H_ANK + 0.017))
    p.add("fabric", upper)
    p.add("fabric", cyl(0.046, (-0.006, HJC_Y, -H_ANK + 0.017), (-0.006, HJC_Y, -0.002)))
    p.add("pad", ring(0.048, 0.040, (-0.006, HJC_Y, -0.010), (-0.006, HJC_Y, 0.004)))
    # stirrup from footplate edge up to the inner hinge knuckle
    p.add("alu", cyl(0.020, (0, ay - 0.010, 0), (0, ay + 0.004, 0)))
    p.add("carbon", bar((0, ay - 0.003, 0), (0.0, ay - 0.003, -H_ANK + 0.016), 0.030, 0.008, fillet=0.004))
    p.add("carbon", box((0.10, 0.012, 0.006), (0.01, ay - 0.008, -H_ANK + 0.015), 0.002))
    # spring lever to the AFO cartridge
    p.add("alu", bar((0, ay + 0.012, 0), (-0.030, ay + 0.012, 0.030), 0.012, 0.008, fillet=0.003))
    return p


# ------------------------------------------------------------- EXA-100 detail
def build_actuator_detail():
    """EXA-100 internals in a local frame: axis along +Y, output on +Y."""
    comps = []  # (name, material, shape)

    def c(name, mat, shape):
        comps.append((name, mat, shape))

    y0 = -W_ACT / 2
    c("housing", "alu", ring(0.050, 0.0475, (0, y0 + 0.004, 0), (0, 0.025, 0)))
    c("housing_accent", "accent", ring(0.0508, 0.0480, (0, 0.012, 0), (0, 0.018, 0)))
    c("rear_cover", "carbon", cyl(0.047, (0, y0, 0), (0, y0 + 0.004, 0)).cut(cyl(0.008, (0, y0, 0), (0, y0 + 0.004, 0))))
    c("motor_encoder_pcb", "pcb", ring(0.030, 0.010, (0, y0 + 0.004, 0), (0, y0 + 0.0055, 0)))
    c("brake_coil", "copper", ring(0.038, 0.016, (0, -0.0245, 0), (0, -0.0185, 0)))
    c("brake_armature", "steel", ring(0.034, 0.013, (0, -0.0185, 0), (0, -0.0160, 0)))
    c("brake_friction_disc", "rubber", ring(0.032, 0.013, (0, -0.0160, 0), (0, -0.0135, 0)))
    stator = ring(0.043, 0.031, (0, -0.011, 0), (0, 0.007, 0))
    for k in range(24):  # stator slots
        a = 360 * k / 24
        slot = box((0.006, 0.020, 0.003), (0.0335, -0.002, 0)).rotate(V(0, 0, 0), V(0, 1, 0), a)
        stator = stator.cut(slot)
    c("stator", "steel", stator)
    c("end_turns_rear", "copper", ring(0.040, 0.032, (0, -0.0145, 0), (0, -0.011, 0)))
    c("end_turns_front", "copper", ring(0.040, 0.032, (0, 0.007, 0), (0, 0.0105, 0)))
    c("rotor_magnets", "magnet", ring(0.0305, 0.0260, (0, -0.010, 0), (0, 0.006, 0)))
    c("rotor_hub", "alu", ring(0.026, 0.012, (0, -0.004, 0), (0, 0.000, 0)))
    c("hollow_shaft", "steel", ring(0.012, 0.008, (0, -0.028, 0), (0, 0.029, 0)))
    wg = cq.Workplane("XZ").ellipse(0.0250, 0.0228).extrude(-0.008).val().translate(V(0, 0.011, 0))
    c("wave_generator", "steel", wg.cut(cyl(0.012, (0, 0.010, 0), (0, 0.020, 0))))
    flex = ring(0.0280, 0.0262, (0, 0.010, 0), (0, 0.026, 0)).fuse(ring(0.0280, 0.014, (0, 0.024, 0), (0, 0.026, 0)))
    c("flexspline", "alu", flex)
    c("circular_spline", "steel", ring(0.040, 0.0285, (0, 0.010, 0), (0, 0.018, 0)))
    c("xrb_outer", "steel", ring(0.047, 0.0425, (0, 0.018, 0), (0, 0.028, 0)))
    c("xrb_inner", "steel", ring(0.0395, 0.030, (0, 0.018, 0), (0, 0.028, 0)))
    for k in range(28):
        a = 2 * math.pi * k / 28
        x, z = 0.041 * math.cos(a), 0.041 * math.sin(a)
        c(f"xrb_roller_{k}", "alu", cyl(0.0016, (x, 0.0215, z), (x, 0.0245, z)))
    flexure = ring(0.038, 0.033, (0, 0.026, 0), (0, 0.031, 0)).fuse(ring(0.016, 0.009, (0, 0.026, 0), (0, 0.031, 0)))
    for k in range(4):
        spoke = box((0.018, 0.005, 0.007), (0.0245, 0.0285, 0)).rotate(V(0, 0, 0), V(0, 1, 0), 90 * k + 45)
        flexure = flexure.fuse(spoke)
        c(f"strain_gauge_{k}", "copper", box((0.006, 0.0006, 0.004), (0.0245, 0.0313, 0)).rotate(V(0, 0, 0), V(0, 1, 0), 90 * k + 45))
    c("torque_sensor_flexure", "alu", flexure)
    c("output_encoder_ring", "magnet", ring(0.0325, 0.0300, (0, 0.0255, 0), (0, 0.0262, 0)))
    flange = cyl(0.045, (0, 0.031, 0), (0, 0.039, 0)).cut(cyl(0.008, (0, 0.031, 0), (0, 0.039, 0)))
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        flange = flange.cut(cyl(0.0022, (0.036 * math.cos(a), 0.031, 0.036 * math.sin(a)), (0.036 * math.cos(a), 0.040, 0.036 * math.sin(a))))
    c("output_flange", "alu", flange)
    c("output_cap", "accent", ring(0.016, 0.008, (0, 0.039, 0), (0, 0.0405, 0)))
    return comps


# ---------------------------------------------------------------------- export
def export_part(part, folder, tol=0.0004, ang=0.12, step=True):
    meshes = {}
    for mat, comp in part.compounds().items():
        f = folder / f"{part.name}__{mat}.stl"
        cq.exporters.export(comp, str(f), tolerance=tol, angularTolerance=ang)
        meshes[mat] = f
    if not step:
        return meshes
    asm = cq.Assembly(name=part.name)
    for mat, comp in part.compounds().items():
        asm.add(comp, name=f"{part.name}_{mat}", color=cq.Color(*MATERIALS[mat][0]))
    asm.export(str(folder / f"{part.name}.step"))
    return meshes


def main():
    parts_dir = OUT / "parts"
    act_dir = OUT / "actuator"
    parts_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    pelvis, thigh, shank, foot = build_pelvis(), build_thigh(), build_shank(), build_foot()
    segments = {
        "pelvis": pelvis,
        "thigh_L": thigh, "shank_L": shank, "foot_L": foot,
        "thigh_R": thigh.mirrored("thigh_R"),
        "shank_R": shank.mirrored("shank_R"),
        "foot_R": foot.mirrored("foot_R"),
    }
    manifest = {"materials": {k: {"rgb": v[0], "metallic": v[1], "roughness": v[2]} for k, v in MATERIALS.items()},
                "segments": {}, "geometry": {}}
    for name, part in segments.items():
        meshes = export_part(part, parts_dir)
        manifest["segments"][name] = {"files": {m: f.name for m, f in meshes.items()}}
        print(f"  {name:8s} {len(meshes)} materials")

    # standing-pose assembly (all joint angles zero)
    hip_z = H_ANK + L_SH + L_TH
    origin = {"pelvis": (0, 0, hip_z), "thigh_L": (0, 0, hip_z), "thigh_R": (0, 0, hip_z),
              "shank_L": (0, 0, hip_z - L_TH), "shank_R": (0, 0, hip_z - L_TH),
              "foot_L": (0, 0, H_ANK), "foot_R": (0, 0, H_ANK)}
    asm = cq.Assembly(name="Volition")
    for name, part in segments.items():
        for mat, comp in part.compounds().items():
            asm.add(comp, name=f"{name}_{mat}", color=cq.Color(*MATERIALS[mat][0]),
                    loc=cq.Location(V(*origin[name])))
    asm.export(str(OUT / "Volition_standing.step"))
    manifest["geometry"] = {"thigh_len": L_TH, "shank_len": L_SH, "ankle_height": H_ANK,
                            "hip_height_standing": hip_z, "segment_origins_standing": origin,
                            "hip_act_y": STRUCT.hip_act_y, "knee_act_y": STRUCT.knee_act_y,
                            "ankle_hinge_y": STRUCT.ankle_hinge_y, "hjc_y": HJC_Y}

    # actuator detail: full, quarter-section cut-away, exploded
    comps = build_actuator_detail()
    act_asm = cq.Assembly(name="EXA-100")
    for name, mat, s in comps:
        act_asm.add(s, name=name, color=cq.Color(*MATERIALS[mat][0]))
    act_asm.export(str(OUT / "EXA-100_actuator.step"))
    cutter = box((0.2, 0.2, 0.2), (0.1, 0.0, 0.1))
    explode = [("rear_cover", "motor_encoder_pcb"), ("brake_",), ("stator", "end_turns"), ("rotor_",),
               ("hollow_shaft",), ("wave_generator",), ("flexspline",), ("circular_spline",), ("xrb_",),
               ("torque_sensor", "strain_gauge", "output_encoder"), ("output_",)]
    by_view = {"cutaway": {}, "exploded": {}}
    for name, mat, s in comps:
        by_view["cutaway"].setdefault(mat, []).append(s.cut(cutter))
        group = next((k for k, g in enumerate(explode) if name.startswith(g)), None)
        if group is not None:  # housing is left out of the exploded view
            by_view["exploded"].setdefault(mat, []).append(s.translate(V(0, 0.030 * (group - len(explode) / 2), 0)))
    for view, mats in by_view.items():
        for mat, shapes in mats.items():
            cq.exporters.export(cq.Compound.makeCompound(shapes), str(act_dir / f"{view}__{mat}.stl"),
                                tolerance=0.0002, angularTolerance=0.1)
    manifest["actuator_components"] = [n for n, _, _ in comps if not n[-1].isdigit()]

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    build_glb(manifest, parts_dir, OUT / "volition.glb")
    # coarser tessellation for the web viewer
    web_dir = OUT / "parts_web"
    web_dir.mkdir(exist_ok=True)
    for part in segments.values():
        export_part(part, web_dir, tol=0.0012, ang=0.35, step=False)
    build_glb(manifest, web_dir, OUT / "volition_web.glb", decimate=0.6)
    print("wrote", OUT)


def build_glb(manifest, parts_dir, path, decimate=0.0):
    """One node per segment/material, in segment-local frames, for the web viewer.
    decimate: fraction of triangles to remove on meshes above 4k faces."""
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    for seg, info in manifest["segments"].items():
        for mat, fname in info["files"].items():
            m = trimesh.load(parts_dir / fname, force="mesh")
            m.merge_vertices()
            if decimate and len(m.faces) > 4000:
                m = m.simplify_quadric_decimation(percent=decimate)
            rgb, metal, rough = manifest["materials"][mat]["rgb"], manifest["materials"][mat]["metallic"], manifest["materials"][mat]["roughness"]
            emissive = rgb if mat == "led" else (0, 0, 0)
            lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]  # glTF factors are linear
            m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                name=mat, baseColorFactor=np.array([*lin, 1.0]), metallicFactor=metal,
                roughnessFactor=rough, emissiveFactor=np.array(emissive)))
            scene.add_geometry(m, node_name=f"{seg}__{mat}", geom_name=f"{seg}__{mat}")
    path.write_bytes(scene.export(file_type="glb"))


if __name__ == "__main__":
    main()
