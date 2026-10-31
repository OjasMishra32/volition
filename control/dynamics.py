"""
Planar (sagittal) Newton-Euler inverse dynamics of user + exoskeleton.

Bodies: HAT (head-arms-trunk + pelvis frame + backpack + hip actuator housings),
and per leg thigh (+ thigh link + knee actuator), shank (+ shank link), foot
(+ AFO footplate).  A complete-SCI user contributes no muscle torque, so the
hip and knee actuators must supply the full joint moments computed here.

External supports on the HAT: forearm crutches carry `crutch_c` of the vertical
load (applied through the HAT COM by default, i.e. they do not balance the trunk;
`crutch_at_hands=True` applies it at the handles instead) and a
chair carries `chair` of it before seat-off (plus the thigh weight).  Swing legs
are solved distal -> proximal; the remaining hip wrench is split between the
stance legs by `share_L` and stance legs are solved proximal -> distal.

Planar convention: vectors are (x, z); moments and planar angles are about +Y.
Joint torques are returned in the joint-flexion convention of params.py
(positive torque increases hip flexion / knee flexion / ankle dorsiflexion).
The stance ankle moment and centre of pressure are reported so the passive-AFO
assumption can be checked; `J` is the inertia each joint sees with the others locked.
"""
from dataclasses import dataclass

import numpy as np

from params import USER, STRUCT, ACT, G
from control.gait import DT, L1, L2, limb

GV = np.array([0.0, -G])


@dataclass
class Body:
    m_hat: float
    c_hat: np.ndarray     # trunk frame (x forward, z up) from the hip midpoint
    i_hat: float
    m_th: float
    c_th: float           # distance from the proximal joint along the segment
    i_th: float
    m_sh: float
    c_sh: float
    i_sh: float
    m_ft: float
    c_ft: np.ndarray      # foot frame (x forward, z up) from the ankle
    i_ft: float
    total: float
    user_mass: float


def _combine(parts):
    """[(mass, (x, z), own inertia about own COM)] -> mass, COM, inertia about COM."""
    m = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1], float) for p in parts) / m
    i = sum(p[2] + p[0] * float(np.sum((np.asarray(p[1], float) - c) ** 2)) for p in parts)
    return m, c, i


def make_body(user_mass=USER.mass):
    M = user_mass
    lf = USER.foot_len
    bp = STRUCT.backpack_center
    bd, _, bh = STRUCT.backpack_size
    m_h = USER.hat_mass_frac * M
    hat = _combine([
        (m_h, (0.0, USER.hat_com_above_hip), m_h * USER.hat_rgyr ** 2),
        (STRUCT.m_pelvis_frame, (-0.05, 0.05), STRUCT.m_pelvis_frame * 0.12 ** 2),
        (STRUCT.m_backpack, (bp[0], bp[2]), STRUCT.m_backpack * (bd ** 2 + bh ** 2) / 12),
        (2 * ACT.mass, (0.0, 0.0), 2 * ACT.mass * 0.03 ** 2),
    ])
    m_t = USER.thigh_mass_frac * M
    th = _combine([
        (m_t, (0.0, -USER.thigh_com_frac * L1), m_t * (USER.thigh_rgyr_frac * L1) ** 2),
        (STRUCT.m_thigh_link, (0.0, -0.5 * L1), STRUCT.m_thigh_link * L1 ** 2 / 12),
        (ACT.mass, (0.0, -L1), ACT.mass * 0.03 ** 2),
    ])
    m_s = USER.shank_mass_frac * M
    sh = _combine([
        (m_s, (0.0, -USER.shank_com_frac * L2), m_s * (USER.shank_rgyr_frac * L2) ** 2),
        (STRUCT.m_shank_link, (0.0, -0.5 * L2), STRUCT.m_shank_link * L2 ** 2 / 12),
    ])
    m_f = USER.foot_mass_frac * M
    ft = _combine([
        (m_f, (0.05, -0.035), m_f * (0.475 * lf) ** 2),
        (STRUCT.m_foot, (0.05, -0.055), STRUCT.m_foot * 0.08 ** 2),
    ])
    total = hat[0] + 2 * (th[0] + sh[0] + ft[0])
    return Body(hat[0], hat[1], hat[2], th[0], -th[1][1], th[2], sh[0], -sh[1][1], sh[2],
                ft[0], ft[1], ft[2], total, M)


def rot(theta, v):
    """Rotate planar (x, z) vectors by theta about +Y."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def cross(r, f):
    """Y component of r x f for planar (x, z) vectors."""
    return r[..., 1] * f[..., 0] - r[..., 0] * f[..., 1]


def _acc(x):
    if len(x) < 5:
        return np.zeros_like(x)
    return np.gradient(np.gradient(x, DT, axis=0), DT, axis=0)


def inverse_dynamics(seg, body, crutch_at_hands=False):
    n = seg['n']
    H, p = seg['pelvis'], seg['trunk']
    th, sh, psi = seg['phi_th'], seg['phi_sh'], seg['psi']
    contact, chair, cc = seg['contact'], seg['chair'], seg['crutch_c']
    b = body
    C_h = H + rot(p, np.broadcast_to(b.c_hat, (n, 2)))
    a_h, al_h = _acc(C_h), _acc(p)
    seat_th = np.stack([np.zeros(n), chair * b.m_th * G], -1)   # chair under the thighs

    legs = []
    for leg in (0, 1):
        K = H + L1 * limb(th[leg])
        A = K + L2 * limb(sh[leg])
        C_t = H + b.c_th * limb(th[leg])
        C_s = K + b.c_sh * limb(sh[leg])
        C_f = A + rot(-psi[leg], np.broadcast_to(b.c_ft, (n, 2)))
        g = dict(K=K, A=A, C_t=C_t, C_s=C_s, C_f=C_f,
                 a_t=_acc(C_t), a_s=_acc(C_s), a_f=_acc(C_f),
                 al_t=-_acc(th[leg]), al_s=-_acc(sh[leg]), al_f=-_acc(psi[leg]))
        # swing solution (foot free), distal -> proximal
        F_sf = b.m_ft * (g['a_f'] - GV)
        M_sf = b.i_ft * g['al_f'] - cross(A - C_f, F_sf)
        F_ts = b.m_sh * (g['a_s'] - GV) + F_sf
        M_ts = b.i_sh * g['al_s'] + M_sf - cross(K - C_s, F_ts) + cross(A - C_s, F_sf)
        F_pt = b.m_th * (g['a_t'] - GV) + F_ts - seat_th
        M_pt = b.i_th * g['al_t'] + M_ts - cross(H - C_t, F_pt) + cross(K - C_t, F_ts)
        g.update(sw_F_pt=F_pt, sw_M_pt=M_pt, sw_M_ts=M_ts, sw_M_sf=M_sf)
        legs.append(g)

    swing = ~contact
    F_sw = sum(swing[l][:, None] * legs[l]['sw_F_pt'] for l in (0, 1))
    M_sw = sum(swing[l] * legs[l]['sw_M_pt'] for l in (0, 1))
    V = b.m_hat * (a_h[:, 1] + G) + F_sw[:, 1]          # vertical support the HAT needs
    F_chair = np.stack([np.zeros(n), chair * V], -1)
    F_crutch = np.stack([np.zeros(n), cc * (1 - chair) * V], -1)
    if crutch_at_hands:
        # crutch reaction enters the trunk at the handles (load-weighted over planted crutches)
        cw = seg['crutch_contact'].astype(float)
        cw /= np.maximum(cw.sum(0), 1e-9)
        hx = (cw * seg['hand'][:, :, 0]).sum(0)
        r_cr = np.stack([hx, C_h[:, 1]], -1)
    else:
        r_cr = C_h
    F_st = F_crutch + F_chair + b.m_hat * (GV - a_h) - F_sw          # sum over stance hips
    M_st = (-b.i_hat * al_h - M_sw + cross(H - C_h, -(F_st + F_sw))
            + cross(H - C_h, F_chair)                                 # chair acts under the hip
            + cross(r_cr - C_h, F_crutch))

    wL = seg['share_L'] * contact[0]
    wR = (1 - seg['share_L']) * contact[1]
    tot = np.maximum(wL + wR, 1e-9)
    w = (wL / tot, wR / tot)

    tau = np.zeros((n, 4))
    grf = np.zeros((2, n))
    ankle_m = np.zeros((2, n))
    cop = np.full((2, n), np.nan)
    J = np.zeros((n, 4))
    I_hat_hip = b.i_hat + b.m_hat * np.sum((C_h - H) ** 2, -1)
    for leg in (0, 1):
        g = legs[leg]
        K, A, C_t, C_s, C_f = g['K'], g['A'], g['C_t'], g['C_s'], g['C_f']
        F_pt = w[leg][:, None] * F_st
        M_pt = w[leg] * M_st
        F_ts = F_pt + seat_th - b.m_th * (g['a_t'] - GV)
        M_ts = M_pt + cross(H - C_t, F_pt) - cross(K - C_t, F_ts) - b.i_th * g['al_t']
        F_sf = F_ts - b.m_sh * (g['a_s'] - GV)
        M_sf = M_ts + cross(K - C_s, F_ts) - cross(A - C_s, F_sf) - b.i_sh * g['al_s']
        GRF = b.m_ft * (g['a_f'] - GV) - F_sf
        st = contact[leg]
        # centre of pressure on the ground from the foot's moment balance
        num = M_sf + cross(A - C_f, F_sf) - C_f[:, 1] * GRF[:, 0] - b.i_ft * g['al_f']
        cop[leg] = np.where(st & (GRF[:, 1] > 20), C_f[:, 0] + num / np.maximum(GRF[:, 1], 1e-6), np.nan)
        M_pt = np.where(st, M_pt, g['sw_M_pt'])
        M_ts = np.where(st, M_ts, g['sw_M_ts'])
        M_sf = np.where(st, M_sf, g['sw_M_sf'])
        tau[:, 2 * leg] = -M_pt
        tau[:, 2 * leg + 1] = M_ts
        ankle_m[leg] = -M_sf
        grf[leg] = np.where(st, GRF[:, 1], 0.0)

        # effective inertia seen by each joint with the other joints locked
        def about(P, pts):
            return sum(i + m * np.sum((c - P) ** 2, -1) for m, c, i in pts)
        swing_hip = about(H, [(b.m_th, C_t, b.i_th), (b.m_sh, C_s, b.i_sh), (b.m_ft, C_f, b.i_ft)])
        swing_knee = about(K, [(b.m_sh, C_s, b.i_sh), (b.m_ft, C_f, b.i_ft)])
        other_leg = b.m_th + b.m_sh + b.m_ft
        stance_hip = w[leg] * I_hat_hip
        stance_knee = w[leg] * (about(K, [(b.m_th, C_t, b.i_th), (b.m_hat, C_h, b.i_hat)])
                                + other_leg * np.sum((H - K) ** 2, -1))
        J[:, 2 * leg] = np.where(st, np.maximum(stance_hip, swing_hip), swing_hip)
        J[:, 2 * leg + 1] = np.where(st, np.maximum(stance_knee, swing_knee), swing_knee)

    return dict(tau=tau, grf=grf, ankle_moment=ankle_m, crutch_load=F_crutch[:, 1],
                chair_load=F_chair[:, 1], J=J, cop_x=cop)
