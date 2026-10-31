#!/usr/bin/env python
"""
Closed-loop Volition scenario at 1 kHz.

  synthetic EEG -> BCI decoder (16 Hz) -> supervisor (+ crutch-button confirm)
  -> task-space planner -> 4 joint controllers -> saturating EXA-100 actuators
  -> linearised human+exo error dynamics, with battery, bus and winding thermals.

Scenario: sit -> FEET_MI -> stand up -> FEET_MI -> walk -> HANDS_MI -> stop
-> HANDS_MI -> sit down.  The decoder decides when each intent is detected; the
scripted user only decides what to imagine and presses the crutch button.

Also runs the low-battery sit-to-stand anti-windup study.

Run:  python exo/control/simulate.py        (writes exo/sim/out/)
"""
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                      # noqa: E402

from params import ACT, BATT, CTRL, EEG_CFG, GAIT, JOINTS, USER, G   # noqa: E402
from control import actuator, bci, dynamics, gait                   # noqa: E402
from control.joint_controller import JointController, JointPlant, N2JM   # noqa: E402
from control.supervisor import (Supervisor, STATE_NAMES, SITTING, STANDING_UP, STANDING,   # noqa: E402
                                WALKING, STOPPING, SITTING_DOWN, FAULT_HOLD)

OUT = ROOT / "sim" / "out"
DT = gait.DT
REC_EVERY = 5                       # 200 Hz recording
N_MAX_STEPS = 16
JOINT_NAMES = {"hip_L": "Hip L", "knee_L": "Knee L", "hip_R": "Hip R", "knee_R": "Knee R"}
SEED = 7


class Reference:
    """Active planned segment with its inverse dynamics (true and model bodies)."""

    def __init__(self, body_true, body_model):
        self.bt, self.bm = body_true, body_model
        self.seg, self.i, self.t0 = None, 0, 0.0

    def load(self, seg):
        idt = dynamics.inverse_dynamics(seg, self.bt)
        idm = dynamics.inverse_dynamics(seg, self.bm)
        seg.update(tau_id=idt["tau"], tau_ff=idm["tau"], J=idt["J"], grf=idt["grf"],
                   crutch_load=idt["crutch_load"], ankle_m=idt["ankle_moment"])
        return seg

    def start(self, seg, t):
        self.seg, self.i, self.t0 = self.load(seg), 0, t

    @property
    def k(self):
        return min(self.i, self.seg["n"] - 1)

    @property
    def finished(self):
        return self.seg["n"] > 1 and self.i >= self.seg["n"] - 1


def freeze(seg, i):
    """Static copy of sample i (FAULT_HOLD: brakes on, hold the current pose)."""
    h = gait._empty(1)
    h["pelvis"][:], h["trunk"][:] = seg["pelvis"][i], seg["trunk"][i]
    for key in ("share_L", "chair", "crutch_c"):
        h[key][:] = seg[key][i]
    for key in ("ankle", "psi", "contact", "crutch_tip", "crutch_contact"):
        h[key][:, 0] = seg[key][:, i]
    h["kind"] = "hold"
    return gait.finalize(h)


class ScriptedUser:
    """What the user imagines and when they press the crutch button."""
    PLAN = [("STAND", "FEET_MI", SITTING, 3.0), ("WALK", "FEET_MI", STANDING, 2.0),
            ("STOP", "HANDS_MI", WALKING, 5.0), ("SIT", "HANDS_MI", STANDING, 1.5)]

    def __init__(self, rng, log):
        self.rng, self.log = rng, log
        self.g, self.since, self.imagining, self.press_at = 0, None, False, None
        self.started = []

    def goal(self):
        return self.PLAN[self.g] if self.g < len(self.PLAN) else None

    def update(self, t, state):
        goal = self.goal()
        if goal is None:
            return
        _, cls, need, wait = goal
        if state != need:
            self.since, self.imagining = None, False
            return
        if self.since is None:
            self.since = t
        if not self.imagining and self.press_at is None and t - self.since >= wait:
            self.imagining = True
            self.started.append(t)
            word = "feet" if cls == "FEET_MI" else "both hands"
            self.log.append((round(t, 3), f"User imagines moving {word}"))

    def mental(self):
        goal = self.goal()
        return goal[1] if goal and self.imagining else "REST"

    def on_detect(self, t, cls):
        goal = self.goal()
        if goal and self.imagining and cls == goal[1]:
            self.imagining = False
            self.press_at = t + self.rng.uniform(0.4, 0.8)

    def press_due(self, t):
        if self.press_at is not None and t >= self.press_at:
            self.press_at = None
            return True
        return False

    def on_result(self, t, action):
        goal = self.goal()
        if goal and action == goal[0]:
            self.g += 1
            self.since = None
        elif goal:                       # blocked or expired: try again shortly
            self.since = t - goal[3] + 1.0


def fk_actual(q, ankle, contact, share_L, sh_plan, L1=gait.L1, L2=gait.L2):
    """Render-ready kinematics from the ACTUAL joint angles: anchor the loaded stance
    foot at its planned position and chain up through the hip to the other leg."""
    hip, knee = q[:, [0, 2]], q[:, [1, 3]]
    sols = []
    for a in (0, 1):
        o = 1 - a
        sh_a = sh_plan[:, a]
        th_a = sh_a + knee[:, a]
        p = hip[:, a] - th_a
        K_a = ankle[:, a] - L2 * gait.limb(sh_a)
        H = K_a - L1 * gait.limb(th_a)
        th_o = hip[:, o] - p
        sh_o = th_o - knee[:, o]
        K_o = H + L1 * gait.limb(th_o)
        A_o = K_o + L2 * gait.limb(sh_o)
        th = np.stack([th_a, th_o], 1) if a == 0 else np.stack([th_o, th_a], 1)
        sh = np.stack([sh_a, sh_o], 1) if a == 0 else np.stack([sh_o, sh_a], 1)
        K = np.stack([K_a, K_o], 1) if a == 0 else np.stack([K_o, K_a], 1)
        A = np.stack([ankle[:, a], A_o], 1) if a == 0 else np.stack([A_o, ankle[:, a]], 1)
        sols.append(dict(p=p, H=H, th=th, sh=sh, K=K, A=A))
    use_L = contact[:, 0] & ((share_L >= 0.5) | ~contact[:, 1])
    out = {}
    for key in sols[0]:
        m = use_L.reshape((-1,) + (1,) * (sols[0][key].ndim - 1))
        out[key] = np.where(m, sols[0][key], sols[1][key])
    return out


def run_scenario(seed=SEED, verbose=True):
    t_wall = time.time()
    rng = np.random.default_rng(seed)
    body_true = dynamics.make_body(USER.mass)
    body_model = dynamics.make_body(USER.mass * (1 + CTRL.model_mass_error))
    bw = body_true.total * G

    dec, calib = bci.calibrate(seed=11)
    false_act = bci.false_activation_test(dec, seed=21)
    gen = bci.EEGGenerator(seed + 100)
    dec.reset()

    sup = Supervisor(bw)
    log = sup.log
    user = ScriptedUser(rng, log)
    ref = Reference(body_true, body_model)
    xa = 0.0
    chairs = [float(gait.seated_hip(xa)[0])]
    ref.start(gait.make_hold("sit", xa), 0.0)
    ctrl = JointController(DT)
    plant = JointPlant(DT)
    batt = actuator.Battery(soc=0.62)
    t_w = np.full(4, ACT.t_ambient + 8.0)
    ctrl.reset(ref.seg["tau_id"][0])
    tau_joint = ref.seg["tau_id"][0].copy()

    heel_strikes, step_events = [], []
    last_bci = 0.0
    artifact, pop_scheduled = False, False
    last_sat_t = np.full(4, -9.0)
    prev_reason = np.zeros(4, int)
    sat_time = np.zeros((4, 6))
    walk_steps_total = 0
    t_end = None

    R = {k: [] for k in ("t", "state", "q_ref", "q", "tau_cmd", "tau_out", "tau_joint", "tau_id", "lo", "hi",
                         "reason", "iq", "t_wind", "p_elec", "v_bus", "i_bus", "soc", "chopper", "grf", "crutch_load",
                         "contact", "share_L", "ankle_plan", "sh_plan", "psi", "crutch_tip", "crutch_contact",
                         "hand", "chair_x", "ankle_m", "integ", "chair")}
    E = {k: [] for k in ("t", "disp", "clip")}
    B = {k: [] for k in ("t", "prob", "evid", "valid")}
    latencies = []

    def signals():
        s, k = ref.seg, ref.k
        return dict(v_bus=batt.v_bus, contact_L=bool(s["contact"][0, k]), contact_R=bool(s["contact"][1, k]),
                    grf_L=float(s["grf"][0, k]), grf_R=float(s["grf"][1, k]),
                    crutch_load=float(s["crutch_load"][k]), chair_behind=True,
                    track_err=plant.e, tau_joint=tau_joint, t_wind=t_w, heartbeat_age=t - last_bci)

    def begin(action, t):
        nonlocal xa
        if action == "STAND":
            ref.start(gait.make_sts(xa), t)
        elif action == "SIT":
            ref.start(gait.make_sts(xa, reverse=True), t)
        elif action == "WALK":
            ref.start(gait.make_walk(xa, N_MAX_STEPS), t)
            heel_strikes[:] = [t + st["t_land"] for st in ref.seg["steps"]]
        elif action == "STOP":
            tl = ref.i * DT
            k_now = int(tl // GAIT.step_time)
            n_new = min(k_now + 2, ref.seg["n_steps"])
            if n_new < ref.seg["n_steps"]:
                new = gait.blend_walk(ref.seg, gait.make_walk(xa, n_new), ref.i)
                i = ref.i
                ref.seg = ref.load(new)
                ref.i = i
                heel_strikes[:] = [h for h in heel_strikes if h <= ref.t0 + new["duration"]]
                log.append((round(t, 3), f"Stop: finishing step {k_now + 1}, closing step {n_new}"))

    t, tick = 0.0, 0
    t_max = 120.0
    while t < t_max:
        # ---------------- BCI (16 Hz) ----------------
        if t + 1e-9 >= last_bci + 1.0 / EEG_CFG.update_hz or tick == 0:
            n_new = int(round(t * bci.FS)) - gen.n
            if n_new > 0:
                while heel_strikes and heel_strikes[0] <= t:
                    gen.add_heel_strike(heel_strikes.pop(0))
                moving = sup.state in (STANDING_UP, WALKING, STOPPING, SITTING_DOWN)
                v, clip = gen.generate(n_new, user.mental(), moving, 1.0 / GAIT.step_time)
                disp = dec.push(v, clip)
                E["t"].append((gen.n - n_new + np.arange(n_new)) / bci.FS)
                E["disp"].append(disp)
                E["clip"].append(clip)
                res = dec.update(t)
                B["t"].append(t)
                B["prob"].append(res["probs"])
                B["evid"].append(res["evidence"])
                B["valid"].append(res["valid"])
                if clip.any() and not artifact:
                    log.append((round(t, 3), "EEG: C4 electrode pop - ADC clipped, windows rejected"))
                    artifact = True
                if artifact and res["valid"]:
                    log.append((round(t, 3), "EEG: signal clean again - decoding resumed"))
                    artifact = False
                if res["event"]:
                    cls, p = res["event"]
                    if user.imagining and cls == user.mental():
                        latencies.append(t - user.started[-1])
                    sup.intent(t, cls, p)
                    user.on_detect(t, cls)
            last_bci = t

        user.update(t, sup.state)
        if sup.state == STANDING and not pop_scheduled:
            gen.schedule_pop(t + 0.8, "C4", 0.26)
            pop_scheduled = True

        if user.press_due(t):
            action = sup.button(t, signals())
            if action:
                begin(action, t)
            user.on_result(t, action)

        act = sup.tick(t, DT, signals())
        if act == "STOP":
            begin("STOP", t)
        elif act == "SIT":
            begin("SIT", t)
        elif act == "FAULT":
            ref.start(freeze(ref.seg, ref.k), t)

        # ---------------- joint control ----------------
        s, k = ref.seg, ref.k
        moving_ref = s["n"] > 1
        q_ref, qd_ref, qdd_ref = s["q"][k], s["qd"][k], s["qdd"][k]
        if not moving_ref:
            qd_ref, qdd_ref = np.zeros(4), np.zeros(4)
        q = q_ref + plant.e
        qd = qd_ref + plant.ed
        j_seg = s["J"][k]
        vb, tw = batt.v_bus, t_w.copy()
        c = ctrl.step(q_ref, qd_ref, qdd_ref, s["tau_ff"][k], q, qd, j_seg + N2JM,
                      lambda cmd, prev: actuator.saturate(cmd, prev, qd, vb, tw, DT))
        tau_joint = plant.step(c["tau_out"], s["tau_id"][k], qd_ref, qdd_ref, j_seg)
        iq, p_el, p_cu = actuator.electrical(c["tau_out"], qd)
        t_w = actuator.thermal_step(t_w, p_cu, DT)
        batt.step(float(p_el.sum()) + actuator.P_AUX, DT)

        reason = c["reason"].astype(int)
        for j in range(4):
            sat_time[j, reason[j]] += DT
            if 1 <= reason[j] <= 4 and prev_reason[j] == 0 and t - last_sat_t[j] > 1.0:
                log.append((round(t, 3), f"{JOINT_NAMES[JOINTS[j]]} torque limit ({actuator.REASONS[reason[j]]})"))
                last_sat_t[j] = t
        prev_reason = reason

        if tick % REC_EVERY == 0:
            R["t"].append(t)
            R["state"].append(sup.state)
            R["q_ref"].append(q_ref)
            R["q"].append(q)
            for key in ("tau_cmd", "tau_out", "lo", "hi", "integ"):
                R[key].append(c[key])
            R["reason"].append(reason)
            R["tau_joint"].append(tau_joint)
            R["tau_id"].append(s["tau_id"][k])
            R["iq"].append(iq)
            R["t_wind"].append(t_w.copy())
            R["p_elec"].append(p_el)
            R["v_bus"].append(batt.v_bus)
            R["i_bus"].append(batt.i_batt)
            R["soc"].append(batt.soc)
            R["chopper"].append(batt.chopper)
            R["grf"].append(s["grf"][:, k])
            R["crutch_load"].append(s["crutch_load"][k])
            R["contact"].append(s["contact"][:, k])
            R["share_L"].append(s["share_L"][k])
            R["ankle_plan"].append(s["ankle"][:, k])
            R["sh_plan"].append(s["phi_sh"][:, k])
            R["psi"].append(s["psi"][:, k])
            R["crutch_tip"].append(s["crutch_tip"][:, k])
            R["crutch_contact"].append(s["crutch_contact"][:, k])
            R["hand"].append(s["hand"][:, k])
            R["chair_x"].append(chairs[-1])
            R["ankle_m"].append(s["ankle_m"][:, k])
            R["chair"].append(s["chair"][k])

        # ---------------- planner progress ----------------
        if moving_ref:
            ref.i += 1
            if s.get("kind") == "walk":
                tl = ref.i * DT
                for st in s["steps"]:
                    if abs(tl - st["t_lift"]) < DT / 2:
                        step_events.append((t, st["k"] + 1, "LR"[st["leg"]]))
            if ref.finished:
                kind = s.get("kind")
                if kind == "walk":
                    xa = s["x_final"]
                    walk_steps_total = s["n_steps"]
                    chairs.append(float(gait.seated_hip(xa)[0]))
                sup.motion_done(t)
                ref.start(gait.make_hold("sit" if kind == "sit_down" else "stand", xa), t)
                if kind == "sit_down":
                    t_end = t + 2.5
        if t_end is not None and t >= t_end:
            break
        if sup.state == FAULT_HOLD and t_end is None:
            t_end = t + 2.0
        t += DT
        tick += 1

    # ---------------- package ----------------
    rec = {k: np.array(v) for k, v in R.items()}
    fk = fk_actual(rec["q"], rec["ankle_plan"], rec["contact"], rec["share_L"], rec["sh_plan"])
    cont = rec["contact"]
    flat = np.abs(rec["psi"]) < 1e-3
    slip = sink = 0.0
    for leg in (0, 1):
        m = cont[:, leg] & flat[:, leg]
        slip = max(slip, float(np.abs(fk["A"][m, leg, 0] - rec["ankle_plan"][m, leg, 0]).max(initial=0)))
        sink = max(sink, float(np.abs(fk["A"][m, leg, 1] - gait.HA).max(initial=0)))
    eeg_t = np.concatenate(E["t"])
    out = dict(
        t=rec["t"], state=rec["state"].astype(np.int8), state_names=np.array(STATE_NAMES),
        joints=np.array(JOINTS),
        trunk_pitch=fk["p"], pelvis_xz=fk["H"],
        phi_thigh_L=fk["th"][:, 0], phi_thigh_R=fk["th"][:, 1],
        phi_shank_L=fk["sh"][:, 0], phi_shank_R=fk["sh"][:, 1],
        psi_foot_L=rec["psi"][:, 0], psi_foot_R=rec["psi"][:, 1],
        ankle_xz_L=fk["A"][:, 0], ankle_xz_R=fk["A"][:, 1],
        knee_xz_L=fk["K"][:, 0], knee_xz_R=fk["K"][:, 1],
        q_ref=rec["q_ref"], q=rec["q"], tau_cmd=rec["tau_cmd"], tau_out=rec["tau_out"],
        tau_joint=rec["tau_joint"], tau_demand=rec["tau_id"],
        tau_lim_pos=rec["hi"], tau_lim_neg=rec["lo"], sat_reason=rec["reason"].astype(np.int8),
        sat_reason_names=np.array(actuator.REASONS), integrator=rec["integ"],
        i_q=rec["iq"], T_wind=rec["t_wind"], p_elec=rec["p_elec"],
        v_bus=rec["v_bus"], i_bus=rec["i_bus"], soc=rec["soc"], chopper_on=rec["chopper"],
        grf_L=rec["grf"][:, 0], grf_R=rec["grf"][:, 1], crutch_load=rec["crutch_load"],
        contact_L=cont[:, 0], contact_R=cont[:, 1],
        crutch_contact_L=rec["crutch_contact"][:, 0], crutch_contact_R=rec["crutch_contact"][:, 1],
        crutch_tip_xyz_L=rec["crutch_tip"][:, 0], crutch_tip_xyz_R=rec["crutch_tip"][:, 1],
        hand_xyz_L=rec["hand"][:, 0], hand_xyz_R=rec["hand"][:, 1],
        chair_x=rec["chair_x"], chair_support=rec["chair"], chairs_x=np.array(chairs[:2] if len(chairs) >= 2 else chairs),
        seat_height=GAIT.seat_height,
        ankle_moment_L=rec["ankle_m"][:, 0], ankle_moment_R=rec["ankle_m"][:, 1],
        eeg_t=eeg_t, eeg_display=np.vstack(E["disp"]).astype(np.float32), eeg_display_ch=np.array(bci.DISPLAY_CH),
        eeg_clip=np.concatenate(E["clip"]),
        bci_t=np.array(B["t"]), bci_prob=np.array(B["prob"]), bci_evidence=np.array(B["evid"]),
        bci_valid=np.array(B["valid"]), bci_classes=np.array(bci.CLASSES),
        bci_threshold=EEG_CFG.evidence_threshold,
        event_t=np.array([e[0] for e in log]), event_text=np.array([e[1] for e in log]),
        step_t=np.array([e[0] for e in step_events]), step_n=np.array([e[1] for e in step_events]),
        step_leg=np.array([e[2] for e in step_events]),
        tau_peak_sw=ACT.tau_peak_sw, foot_heel_x=gait.FOOT_HEEL, foot_toe_x=gait.FOOT_TOE,
        ankle_height=gait.HA, thigh_len=gait.L1, shank_len=gait.L2, crutch_len=gait.CRUTCH_LEN,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "run.npz", **out)

    st = rec["state"]
    peaks = {}
    for name, states in (("stand_up", [STANDING_UP]), ("walk", [WALKING, STOPPING]), ("sit_down", [SITTING_DOWN])):
        m = np.isin(st, states)
        if m.any():
            peaks[name] = {JOINTS[j]: dict(peak_torque_Nm=round(float(np.abs(rec["tau_out"][m, j]).max()), 1),
                                           peak_speed_rad_s=round(float(np.abs(np.gradient(rec["q"][:, j], REC_EVERY * DT)[m]).max()), 2),
                                           max_track_err_deg=round(float(np.degrees(np.abs(rec["q_ref"][m, j] - rec["q"][m, j]).max())), 2))
                           for j in range(4)}
    energy_wh = float(np.sum(rec["v_bus"] * rec["i_bus"]) * REC_EVERY * DT / 3600)
    meta = dict(
        scenario="sit -> stand -> walk -> stop -> sit, BCI-commanded with crutch-button confirmation",
        duration_s=round(float(rec["t"][-1]), 2), steps_walked=int(walk_steps_total),
        distance_m=round(float(xa), 3), chairs_x=[round(c, 3) for c in chairs],
        user=dict(height_m=USER.height, mass_kg=USER.mass, system_mass_kg=round(body_true.total, 1)),
        state_timeline=[(round(float(rec["t"][i]), 2), STATE_NAMES[int(st[i])]) for i in
                        np.flatnonzero(np.diff(np.concatenate([[-1], st])) != 0)],
        peaks=peaks,
        saturation_time_s={JOINTS[j]: {actuator.REASONS[r]: round(float(sat_time[j, r]), 3)
                                       for r in range(1, 6) if sat_time[j, r] > 0} for j in range(4)},
        battery=dict(soc_start=0.62, soc_end=round(float(batt.soc), 4), v_bus_min=round(float(rec["v_bus"].min()), 2),
                     v_bus_max=round(float(rec["v_bus"].max()), 2), energy_Wh=round(energy_wh, 2),
                     chopper_used=bool(rec["chopper"].any())),
        winding_temp_max_C=round(float(rec["t_wind"].max()), 2),
        bci=dict(calibration=calib, false_activation=false_act,
                 detection_latency_s=[round(x, 2) for x in latencies],
                 windows_rejected_frac=round(1 - float(np.mean(B["valid"])), 3)),
        feet_check=dict(max_slip_m=round(slip, 4), max_sink_or_float_m=round(sink, 4)),
        ankle_moment_peak_Nm=round(float(np.abs(rec["ankle_m"]).max()), 1),
        fault=sup.fault, events=len(log), wall_time_s=round(time.time() - t_wall, 1),
    )
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(json.dumps({k: meta[k] for k in ("duration_s", "steps_walked", "distance_m", "state_timeline",
                                               "saturation_time_s", "battery", "feet_check", "fault",
                                               "wall_time_s")}, indent=1))
        print("BCI latency", meta["bci"]["detection_latency_s"], "acc", round(calib["heldout_window_accuracy"], 3))
        for e in log:
            print(f"  {e[0]:7.2f}  {e[1]}")
    return out, meta


# ------------------------------------------------------------------ anti-windup study
def sts_study(antiwindup, ocv=41.6, seed=0):
    body_true = dynamics.make_body(USER.mass)
    body_model = dynamics.make_body(USER.mass * (1 + CTRL.model_mass_error))
    ref = Reference(body_true, body_model)
    segs = [gait.make_hold("sit", 0.0, 0.5), gait.make_sts(0.0), gait.make_hold("stand", 0.0, 3.0)]
    ctrl = JointController(DT, antiwindup=antiwindup)
    plant = JointPlant(DT)
    batt = actuator.Battery(soc=actuator.Battery.soc_for_ocv(ocv))
    t_w = np.full(4, ACT.t_ambient + 8.0)
    rec = {k: [] for k in ("t", "q_ref", "q", "cmd", "out", "lo", "hi", "integ", "reason", "v")}
    t = 0.0
    sts_window = None
    for si, seg in enumerate(segs):
        ref.start(seg, t)
        if si == 0:
            ctrl.reset(ref.seg["tau_id"][0])
        if si == 1:
            sts_window = (t, t + gait.GAIT.sts_duration)
        for k in range(ref.seg["n"]):
            s = ref.seg
            q_ref, qd_ref, qdd_ref = s["q"][k], s["qd"][k], s["qdd"][k]
            q, qd = q_ref + plant.e, qd_ref + plant.ed
            vb, tw = batt.v_bus, t_w.copy()
            c = ctrl.step(q_ref, qd_ref, qdd_ref, s["tau_ff"][k], q, qd, s["J"][k] + N2JM,
                          lambda cmd, prev: actuator.saturate(cmd, prev, qd, vb, tw, DT))
            plant.step(c["tau_out"], s["tau_id"][k], qd_ref, qdd_ref, s["J"][k])
            _, p_el, p_cu = actuator.electrical(c["tau_out"], qd)
            t_w = actuator.thermal_step(t_w, p_cu, DT)
            batt.step(float(p_el.sum()) + actuator.P_AUX, DT)
            for key, val in (("t", t), ("q_ref", q_ref), ("q", q), ("cmd", c["tau_cmd"]), ("out", c["tau_out"]),
                             ("lo", c["lo"]), ("hi", c["hi"]), ("integ", c["integ"]), ("reason", c["reason"]),
                             ("v", batt.v_bus)):
                rec[key].append(val)
            t += DT
    rec = {k: np.array(v) for k, v in rec.items()}
    rec["sts_window"] = sts_window
    return rec


def study_metrics(r, j):
    t = r["t"]
    e = np.degrees(r["q_ref"][:, j] - r["q"][:, j])
    t1 = r["sts_window"][1]
    sat = (r["reason"][:, j] >= 1) & (r["reason"][:, j] <= 4)
    lag = e.min()          # knee extends late -> q above q_ref -> e negative
    i_lag = int(np.argmin(e))
    over = float(max(e[i_lag:].max(), 0.0))
    bad = np.flatnonzero(np.abs(e) > 0.5)
    settle = float(max(t[bad[-1]] - t1, 0.0)) if len(bad) else 0.0
    return dict(peak_error_deg=round(float(np.abs(e).max()), 2), max_lag_deg=round(float(-lag), 2),
                overshoot_deg=round(over, 2), settling_after_sts_s=round(settle, 3),
                time_in_saturation_s=round(float(sat.sum() * DT), 3),
                integrator_peak_Nm=round(float(np.abs(r["integ"][:, j]).max()), 1))


def run_antiwindup_study(ocv=41.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    naive = sts_study(False, ocv)
    aw = sts_study(True, ocv)
    j = 3          # knee_R (both knees are symmetric)
    res = dict(case=f"sit-to-stand, 85 kg user, battery OCV {ocv} V (low-voltage torque derate active)",
               joint=JOINTS[j], v_bus_range=[round(float(aw["v"].min()), 2), round(float(aw["v"].max()), 2)],
               torque_limit_min_Nm=round(float(aw["hi"][:, j].min()), 1),
               naive_pid=study_metrics(naive, j), back_calculation=study_metrics(aw, j),
               hip_naive=study_metrics(naive, 2), hip_back_calculation=study_metrics(aw, 2),
               note="Joint-level study without the supervisor; the bus dips below the 40 V forced-sit "
                    "threshold under seat-off load, so the full system would abort to a guided sit.")
    (OUT / "antiwindup.json").write_text(json.dumps(res, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    t = aw["t"]
    sat = (aw["reason"][:, j] >= 1) & (aw["reason"][:, j] <= 4)
    for a in ax:
        a.fill_between(t, 0, 1, where=sat, color="#f2c6c6", alpha=0.6, transform=a.get_xaxis_transform(),
                       label="limit active (anti-windup run)")
        a.axvspan(*aw["sts_window"], color="#e8eef8", alpha=0.5, zorder=0)
    ax[0].plot(t, np.degrees(aw["q_ref"][:, j]), color="#222", lw=1.5, ls="--", label="reference")
    ax[0].plot(naive["t"], np.degrees(naive["q"][:, j]), color="#d1495b", lw=1.6, label="naive PID")
    ax[0].plot(t, np.degrees(aw["q"][:, j]), color="#2e86ab", lw=1.6, label="back-calculation anti-windup")
    ax[0].set_ylabel("knee flexion [deg]")
    ax[0].legend(loc="upper right", frameon=False)
    ax[0].set_title(f"Sit-to-stand at low battery (OCV {ocv} V): knee R tracking with actuator saturation")
    ax[1].plot(t, aw["hi"][:, j], color="#888", lw=1, label="torque limits")
    ax[1].plot(t, aw["lo"][:, j], color="#888", lw=1)
    ax[1].plot(naive["t"], naive["cmd"][:, j], color="#d1495b", lw=0.8, ls=":", label="naive: command")
    ax[1].plot(naive["t"], naive["out"][:, j], color="#d1495b", lw=1.5, label="naive: delivered")
    ax[1].plot(t, aw["cmd"][:, j], color="#2e86ab", lw=0.8, ls=":", label="anti-windup: command")
    ax[1].plot(t, aw["out"][:, j], color="#2e86ab", lw=1.5, label="anti-windup: delivered")
    ax[1].set_ylabel("knee torque [N m]")
    lim = max(150, float(np.abs(aw["out"][:, j]).max()) * 1.3)
    ax[1].set_ylim(-lim, lim)
    ax[1].legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    ax[2].plot(naive["t"], naive["integ"][:, j], color="#d1495b", lw=1.5, label="naive")
    ax[2].plot(t, aw["integ"][:, j], color="#2e86ab", lw=1.5, label="anti-windup")
    ax[2].set_ylabel("integrator state [N m]")
    ax[2].set_xlabel("time [s]")
    ax[2].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "antiwindup.png", dpi=150)
    plt.close(fig)
    return res


if __name__ == "__main__":
    s0, s1 = gait.make_hold("stand", 0.0), gait.make_hold("sit", 0.0)
    assert abs(s0["phi_th"][0, 0]) < math.radians(3) and abs(s0["phi_sh"][0, 0]) < math.radians(4), "standing not vertical"
    assert math.radians(80) < s1["q"][0, 1] < math.radians(100), "seated knee should be ~90 deg"
    run_scenario()
    print(json.dumps(run_antiwindup_study(), indent=1))
