"""
Task-space trajectory generator for walking and sit <-> stand transfers.

Plans are built in Cartesian space -- pelvis (hip-joint midpoint), both ankle
points, trunk pitch, crutch tips -- and converted to joint space with 2-link IK,
so planted feet and crutch tips are exactly stationary by construction.
Leg / crutch index 0 = left, 1 = right.  All arrays are sampled at DT.

Segments are plain dicts; the simulator strings them together at runtime as the
supervisor approves transitions (sit -> stand -> walk N -> stop -> sit).
"""
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from params import USER, GAIT, CTRL, STRUCT

DT = 1e-3
L1, L2, HA = USER.thigh_len, USER.shank_len, USER.ankle_height
FOOT_HEEL, FOOT_TOE = -0.22 * USER.foot_len, 0.78 * USER.foot_len

STAND_TRUNK = math.radians(3)
STAND_HIP_DX = 0.02            # hip slightly ahead of the ankles when standing
SEAT_HIP_ABOVE = 0.085         # hip-joint centre above the seat surface
SEATED_SHANK = math.radians(-10)   # feet tucked back under the knees
SEATED_TRUNK = math.radians(5)
MIN_KNEE = math.radians(3)
HIP_FLEX_PLAN_MAX = CTRL.hip_rom[1] - math.radians(3)
CRUTCH_LEN = 0.485 * USER.height   # tip to handle (wrist height)
CRUTCH_Y, HAND_Y = 0.30, 0.25
CRUTCH_AHEAD_STAND, CRUTCH_AHEAD_WALK = 0.20, 0.10
CRUTCH_LIFT = 0.05
SETTLE_T = 0.8
BLEND_T = 0.6
# sit-to-stand hip path: fraction of the seated -> standing hip displacement at each knot
STS_KNOTS = [0.0, 0.30, 0.55, 0.80, 1.0]
STS_FX = [0.0, 0.22, 0.40, 0.80, 1.0]
STS_FZ = [0.0, 0.03, 0.45, 0.92, 1.0]
STS_ARM_PUSH = 0.15            # extra crutch share the user pushes with around seat-off
HEEL_OFF = math.radians(-25)     # trailing-foot pitch at toe-off
SWING_LEAD = 0.5                 # front-loads the swing-foot advance (shank stays upright)
HEEL_OFF_T = 0.55                # heel rise starts just before the lead foot lands
MIN_SOLE_CLEARANCE = 0.02        # mid-swing heel/toe clearance


def smoothstep(x):
    """Quintic 0->1 with zero first and second derivatives at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def bump(x):
    """0 -> 1 -> 0 on [0, 1], zero slope and curvature at the ends."""
    x = np.clip(x, 0.0, 1.0)
    return 64 * x ** 3 * (1 - x) ** 3


def leg_reach(knee):
    return np.sqrt(L1 ** 2 + L2 ** 2 + 2 * L1 * L2 * np.cos(knee))


def limb(phi):
    """Unit vector (x, z) of a limb segment hanging at absolute angle phi."""
    return np.stack([np.sin(phi), -np.cos(phi)], -1)


def ik(hip, ankle):
    """Hip and ankle points (..., 2) -> (phi_thigh, phi_shank), knee bending forward."""
    d = ankle - hip
    D = np.minimum(np.hypot(d[..., 0], d[..., 1]), leg_reach(math.radians(0.5)))
    alpha = np.arctan2(d[..., 0], -d[..., 1])
    beta = np.arccos(np.clip((L1 ** 2 + D ** 2 - L2 ** 2) / (2 * L1 * D), -1, 1))
    knee = np.arccos(np.clip((D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
    return alpha + beta, alpha + beta - knee


def standing_hip(xa):
    D = leg_reach(GAIT.standing_knee)
    return np.array([xa + STAND_HIP_DX, HA + math.sqrt(D ** 2 - STAND_HIP_DX ** 2)])


def seated_hip(xa):
    knee = np.array([xa - L2 * math.sin(SEATED_SHANK), HA + L2 * math.cos(SEATED_SHANK)])
    z = GAIT.seat_height + SEAT_HIP_ABOVE
    return np.array([knee[0] - math.sqrt(L1 ** 2 - (z - knee[1]) ** 2), z])


def _empty(n):
    s = dict(
        pelvis=np.zeros((n, 2)), trunk=np.zeros(n),
        ankle=np.zeros((2, n, 2)), psi=np.zeros((2, n)),
        contact=np.ones((2, n), bool), share_L=np.full(n, 0.5),
        chair=np.zeros(n), crutch_c=np.zeros(n),
        crutch_tip=np.zeros((2, n, 3)), crutch_contact=np.ones((2, n), bool),
    )
    s['ankle'][:, :, 1] = HA
    s['crutch_tip'][0, :, 1], s['crutch_tip'][1, :, 1] = CRUTCH_Y, -CRUTCH_Y
    return s


def _move_crutch(s, side, x0, x1, i0, i1, u):
    tip = s['crutch_tip'][side]
    tip[i0:i1, 0] = x0 + (x1 - x0) * smoothstep(u)
    tip[i0:i1, 2] = CRUTCH_LIFT * bump(u)
    s['crutch_contact'][side, i0:i1] = False
    tip[i1:, 0] = x1


def finalize(s):
    """Task space -> joint space, derivatives, hands on the crutch handles."""
    n = len(s['trunk'])
    s['n'] = n
    th, sh = np.empty((2, n)), np.empty((2, n))
    for leg in (0, 1):
        th[leg], sh[leg] = ik(s['pelvis'], s['ankle'][leg])
    s['phi_th'], s['phi_sh'] = th, sh
    s['knee'] = s['pelvis'][None] + L1 * limb(th)
    p = s['trunk']
    q = np.stack([th[0] + p, th[0] - sh[0], th[1] + p, th[1] - sh[1]], 1)
    s['q'] = q
    if n >= 5:
        s['qd'] = np.gradient(q, DT, axis=0)
        s['qdd'] = np.gradient(s['qd'], DT, axis=0)
    else:
        s['qd'] = np.zeros_like(q)
        s['qdd'] = np.zeros_like(q)
    s['ankle_dorsi'] = s['psi'] - sh
    hand_x = np.maximum(s['pelvis'][:, 0] + 0.15, s['crutch_tip'][:, :, 0].mean(0) - 0.03)
    want = np.stack([hand_x, np.zeros(n), s['pelvis'][:, 1] - 0.05], -1)
    hands = np.empty((2, n, 3))
    for side, y in ((0, HAND_Y), (1, -HAND_Y)):
        w = want.copy()
        w[:, 1] = y
        d = w - s['crutch_tip'][side]
        hands[side] = s['crutch_tip'][side] + CRUTCH_LEN * d / np.linalg.norm(d, axis=1, keepdims=True)
    s['hand'] = hands
    return s


def make_hold(kind, xa, duration=0.0):
    """Static seated ('sit') or standing ('stand') pose with feet at x = xa."""
    n = max(1, int(round(duration / DT)))
    s = _empty(n)
    s['kind'] = kind
    s['ankle'][:, :, 0] = xa
    if kind == 'sit':
        s['pelvis'][:] = seated_hip(xa)
        s['trunk'][:] = SEATED_TRUNK
        s['chair'][:] = 1.0
        s['crutch_c'][:] = GAIT.crutch_support_sts
    else:
        s['pelvis'][:] = standing_hip(xa)
        s['trunk'][:] = STAND_TRUNK
        s['crutch_c'][:] = GAIT.crutch_support_walk
    s['crutch_tip'][:, :, 0] = xa + CRUTCH_AHEAD_STAND
    s['x_feet'] = xa
    return finalize(s)


def make_sts(xa, reverse=False):
    """Sit-to-stand (or stand-to-sit when reverse=True) with feet fixed at xa."""
    T = GAIT.stand_to_sit_duration if reverse else GAIT.sts_duration
    n = int(round(T / DT)) + 1
    t = np.arange(n) * DT
    u = smoothstep(t / T)
    if reverse:
        u = 1.0 - u
    s = _empty(n)
    s['kind'] = 'sit_down' if reverse else 'stand_up'
    s['ankle'][:, :, 0] = xa
    h0, h1 = seated_hip(xa), standing_hip(xa)
    hx = PchipInterpolator(STS_KNOTS, h0[0] + (h1[0] - h0[0]) * np.array(STS_FX))(u)
    hz = PchipInterpolator(STS_KNOTS, h0[1] + (h1[1] - h0[1]) * np.array(STS_FZ))(u)
    s['pelvis'][:, 0] = gaussian_filter1d(hx, 40, mode='nearest')
    s['pelvis'][:, 1] = gaussian_filter1d(hz, 40, mode='nearest')
    peak = GAIT.sts_trunk_lean_peak
    trunk = PchipInterpolator([0.0, 0.15, 0.38, 0.60, 0.85, 1.0],
                              [SEATED_TRUNK, math.radians(15), peak, 0.75 * peak,
                               math.radians(10), STAND_TRUNK])(u)
    # the trunk lean is capped so planned hip flexion stays inside the hip ROM
    th, _ = ik(s['pelvis'], s['ankle'][0])
    cap = HIP_FLEX_PLAN_MAX - th
    trunk = gaussian_filter1d(np.minimum(trunk, cap - math.radians(2.5)), 70, mode='nearest')
    s['trunk'][:] = trunk
    s['chair'][:] = 1.0 - smoothstep((u - 0.22) / 0.16)
    s['crutch_c'][:] = (GAIT.crutch_support_sts + STS_ARM_PUSH * np.exp(-0.5 * ((u - 0.42) / 0.12) ** 2)
                        + (GAIT.crutch_support_walk - GAIT.crutch_support_sts) * smoothstep((u - 0.7) / 0.3))
    s['crutch_tip'][:, :, 0] = xa + CRUTCH_AHEAD_STAND
    s['x_feet'] = xa
    s['duration'] = T
    s['trunk_peak_planned'] = float(trunk.max())
    return finalize(s)


def foot_from_toe(toe_x, psi):
    """Ankle point of a foot pivoting on its toe (heel-off), psi < 0 = heel up."""
    v = rot_xz(-psi, np.array([FOOT_TOE, -HA]))
    return toe_x - v[..., 0], -v[..., 1]


def rot_xz(theta, v):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[0] + s * v[1], -s * v[0] + c * v[1]], -1)


def toe_on_ground_pitch(ankle_z):
    """Most toe-down pitch that keeps the toe on (not through) the ground."""
    R = math.hypot(FOOT_TOE, HA)
    return math.atan2(HA, FOOT_TOE) + np.arcsin(np.clip(-ankle_z / R, -1, 1))


def sole_clearance(ankle, psi):
    """Lowest point of the sole (heel or toe) above the ground."""
    heel = ankle[..., 1] + np.sin(psi) * FOOT_HEEL - np.cos(psi) * HA
    toe = ankle[..., 1] + np.sin(psi) * FOOT_TOE - np.cos(psi) * HA
    return np.minimum(heel, toe)


def make_walk(xa, n_steps, first=1):
    """Exoskeleton gait from standing at xa: first step is half a stride, the last
    one a closing step that brings the feet together.  Stance feet are flat, the
    trailing foot rolls onto its toe (heel-off) before lifting, and in swing the
    passive AFO holds the foot near neutral relative to the shank, landing flat."""
    assert n_steps >= 2
    L, T, f = GAIT.step_length, GAIT.step_time, GAIT.double_support_frac
    dur = n_steps * T + SETTLE_T
    n = int(round(dur / DT)) + 1
    t = np.arange(n) * DT
    s = _empty(n)
    s['kind'] = 'walk'
    ax, az = s['ankle'][:, :, 0], s['ankle'][:, :, 1]
    ax[:] = xa
    s['crutch_tip'][:, :, 0] = xa + CRUTCH_AHEAD_STAND
    feet = [xa, xa]
    crutch = [xa + CRUTCH_AHEAD_STAND] * 2
    way_t, way_x = [0.0], [xa + STAND_HIP_DX]
    steps = []
    swings = []

    def idx(tt):
        return int(round(tt / DT))

    X = first
    for k in range(n_steps):
        X = first if k % 2 == 0 else 1 - first
        Y = 1 - X
        t0, ts, t1 = k * T, k * T + f * T, (k + 1) * T
        target = feet[Y] if k == n_steps - 1 else feet[Y] + L
        if k > 0:      # pelvis passes the double-support midpoints at a steady pace
            way_t.append(t0 + f * T / 2)
            way_x.append(0.5 * (feet[0] + feet[1]))
        i0, i_s, i1 = idx(t0), idx(ts), idx(t1)
        w0 = 0.5 if k == 0 else 0.0
        load_Y = np.concatenate([w0 + (1 - w0) * smoothstep((t[i0:i_s] - t0) / (ts - t0)),
                                 np.ones(i1 - i_s)])
        s['share_L'][i0:i1] = load_Y if Y == 0 else 1 - load_Y
        # heel-off: the trailing foot pivots on its toe before lifting
        psi_lift, x_lift, z_lift = 0.0, feet[X], HA
        if k > 0:
            ih = idx(ts - HEEL_OFF_T)
            v = smoothstep((t[ih:i_s] - (ts - HEEL_OFF_T)) / HEEL_OFF_T)
            ps = HEEL_OFF * v
            hx, hz = foot_from_toe(feet[X] + FOOT_TOE, ps)
            s['psi'][X, ih:i_s], ax[X, ih:i_s], az[X, ih:i_s] = ps, hx, hz
            psi_lift, x_lift, z_lift = HEEL_OFF, hx[-1], hz[-1]
        u = (t[i_s:i1] - ts) / (t1 - ts)
        ax[X, i_s:i1] = x_lift + (target - x_lift) * smoothstep(u + SWING_LEAD * u * (1 - u))
        az[X, i_s:i1] = z_lift + (HA - z_lift) * smoothstep(u) + GAIT.swing_clearance * bump(u)
        s['contact'][X, i_s:i1] = False
        ax[X, i1:] = target
        swings.append((X, i_s, i1, u, psi_lift))
        steps.append(dict(k=k, leg=int(X), t_lift=ts, t_land=t1, x_from=feet[X], x_to=target))
        feet[X] = target
        # 2-point gait: the crutch opposite the swing leg moves with it
        ct = target + (CRUTCH_AHEAD_STAND if k == n_steps - 1 else CRUTCH_AHEAD_WALK)
        _move_crutch(s, Y, crutch[Y], ct, i_s, i1, u)
        crutch[Y] = ct

    # settle: share back to 50/50, square up the remaining crutch
    iN, iC = idx(n_steps * T), idx(n_steps * T + 0.6 * SETTLE_T)
    tt = t[iN:] - n_steps * T
    r = 1 - 0.5 * smoothstep(tt / (0.6 * SETTLE_T))
    s['share_L'][iN:] = r if X == 1 else 1 - r      # the stance leg of the closing step held it all
    _move_crutch(s, X, crutch[X], feet[X] + CRUTCH_AHEAD_STAND, iN, iC,
                 (t[iN:iC] - n_steps * T) / (0.6 * SETTLE_T))
    way_t.append(dur)
    way_x.append(feet[0] + STAND_HIP_DX)
    way_t.insert(1, 0.25)                  # start and finish from rest
    way_x.insert(1, way_x[0])
    way_t.insert(-1, dur - 0.25)
    way_x.insert(-1, way_x[-1])
    px = gaussian_filter1d(PchipInterpolator(way_t, way_x)(t), 120, mode='nearest')

    lean = GAIT.trunk_lean_walk
    s['trunk'][:] = STAND_TRUNK + (lean - STAND_TRUNK) * smoothstep(t / T) * (1 - smoothstep((t - (dur - 1.2)) / 1.2))
    s['crutch_c'][:] = GAIT.crutch_support_walk
    kap = GAIT.standing_knee + (GAIT.stance_knee_target - GAIT.standing_knee) * \
        smoothstep(t / (f * T)) * (1 - smoothstep((t - n_steps * T) / (0.8 * SETTLE_T)))
    Dk, D3 = leg_reach(kap), leg_reach(MIN_KNEE)
    z0, zN = standing_hip(xa)[1], standing_hip(feet[0])[1]
    for _ in range(6):
        # pelvis height: stance knee near the target flexion (inverted-pendulum
        # vault), never straighter than MIN_KNEE on any leg
        zreq = np.full((2, n), np.inf)
        zmax = np.empty((2, n))
        for leg in (0, 1):
            dx = px - ax[leg]
            zr = az[leg] + np.sqrt(np.maximum(Dk ** 2 - dx ** 2, 1e-6))
            zreq[leg] = np.where(s['contact'][leg], zr, np.inf)
            zmax[leg] = az[leg] + np.sqrt(np.maximum(D3 ** 2 - dx ** 2, 1e-6))
        zmax = zmax.min(0)
        z = gaussian_filter1d(np.minimum(zreq.min(0), zmax - 0.01), 60, mode='nearest')
        z = gaussian_filter1d(np.minimum(z, zmax), 15, mode='nearest')
        z += (z0 - z[0]) * (1 - smoothstep(t / 0.5)) + (zN - z[-1]) * (1 - smoothstep((dur - t) / 0.5))
        z = np.minimum(z, zmax)
        s['pelvis'][:, 0], s['pelvis'][:, 1] = px, z
        # swing foot pitch: from heel-off, follow the shank at the AFO neutral, land flat
        lifted = False
        for X, i_s, i1, u, psi_lift in swings:
            _, sh = ik(s['pelvis'][i_s:i1], s['ankle'][X, i_s:i1])
            wa = 1 - smoothstep(u / 0.3)
            wb = smoothstep((u - 0.65) / 0.35)
            ps = wa * psi_lift + (1 - wa - wb) * (sh + STRUCT.ankle_afo_neutral)
            ps = np.maximum(ps, toe_on_ground_pitch(az[X, i_s:i1]))
            s['psi'][X, i_s:i1] = gaussian_filter1d(ps, 25, mode='nearest')
            gap = MIN_SOLE_CLEARANCE * bump(u) - sole_clearance(s['ankle'][X, i_s:i1], s['psi'][X, i_s:i1])
            if gap.max() > 1e-4:
                az[X, i_s:i1] += 1.2 * gap.max() * bump(u)
                lifted = True
        if not lifted:
            break
    s['steps'] = steps
    s['x_feet'] = xa
    s['x_final'] = feet[0]
    s['n_steps'] = n_steps
    s['duration'] = dur
    s['monotone'] = bool(np.all(np.diff(px) > -1e-6))
    return finalize(s)


def blend_walk(old, new, i_now):
    """Switch from plan `old` to `new` at sample i_now without a jump: feet and
    crutches already agree up to the current step, pelvis and trunk are cross-faded."""
    nb = int(BLEND_T / DT)
    for key in ('pelvis', 'trunk'):
        a, b = old[key], new[key]
        m = min(len(b), len(a))
        out = b.copy()
        out[:i_now] = a[:i_now]
        j = np.arange(i_now, min(i_now + nb, m))
        w = smoothstep((j - i_now) / nb)
        if b.ndim == 2:
            w = w[:, None]
        out[j] = (1 - w) * a[j] + w * b[j]
        new[key] = out
    return finalize(new)
