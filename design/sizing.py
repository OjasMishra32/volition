#!/usr/bin/env python
"""
EXA-100 actuator sizing check from the planned motions and inverse dynamics.

For the 85 kg design user and the 100 kg maximum user: per-joint peak / RMS
torque, peak speed and power for walking, sit-to-stand and stand-to-sit; the
torque-speed envelope at full, nominal and low battery voltage with the demand
points overlaid; a thermal check over continuous walking; and a battery runtime
estimate.

Run:  python exo/design/sizing.py      (writes exo/design/out/)
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                       # noqa: E402
import matplotlib                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from params import ACT, BATT, GAIT, USER   # noqa: E402
from control import actuator, dynamics, gait   # noqa: E402

OUT = ROOT / "design" / "out"
VOLTS = {"full 54.6 V": BATT.series * BATT.v_cell_max, "nominal 46.8 V": BATT.v_nominal, "low 40.0 V": BATT.uvlo_sit_v}
TASK_COLORS = {"walk": "#2e86ab", "sit_to_stand": "#e07a1f", "stand_to_sit": "#3c9d5d"}


def demands(mass):
    body = dynamics.make_body(mass)
    tasks = {"walk": gait.make_walk(0.0, 6), "sit_to_stand": gait.make_sts(0.0),
             "stand_to_sit": gait.make_sts(0.0, reverse=True)}
    out = {}
    for name, seg in tasks.items():
        r = dynamics.inverse_dynamics(seg, body)
        out[name] = dict(seg=seg, tau=r["tau"], qd=seg["qd"], ankle=r["ankle_moment"],
                         crutch=r["crutch_load"], body=body)
    return out


def stride_window(seg):
    """One steady stride (steps 3-4) of the walking plan."""
    T = GAIT.step_time
    i0, i1 = int(2 * T / gait.DT), int(4 * T / gait.DT)
    return slice(i0, i1)


def joint_stats(d):
    res = {}
    for task, v in d.items():
        tau, qd = v["tau"], v["qd"]
        sl = stride_window(v["seg"]) if task == "walk" else slice(None)
        p = tau * qd
        res[task] = {}
        for jt, cols in (("hip", [0, 2]), ("knee", [1, 3])):
            res[task][jt] = dict(
                peak_torque_Nm=round(float(np.abs(tau[:, cols]).max()), 1),
                rms_torque_Nm=round(float(np.sqrt(np.mean(tau[sl][:, cols] ** 2))), 1),
                peak_speed_rad_s=round(float(np.abs(qd[:, cols]).max()), 2),
                peak_power_W=round(float(p[:, cols].max()), 1),
                peak_regen_W=round(float(-p[:, cols].min()), 1))
        res[task]["peak_ankle_moment_Nm"] = round(float(np.abs(v["ankle"]).max()), 1)
        res[task]["peak_crutch_load_N"] = round(float(v["crutch"].max()), 1)
    return res


def walking_power(d):
    v = d["walk"]
    sl = stride_window(v["seg"])
    tau, qd = v["tau"][sl], v["qd"][sl]
    iq, p, p_cu = actuator.electrical(tau, qd)
    p_bus = float(p.sum(1).mean()) + actuator.P_AUX
    p_bus_no_regen = float(np.maximum(p.sum(1), 0).mean()) + actuator.P_AUX
    iq_rms = np.sqrt(np.mean(iq ** 2, 0))
    t_ss = ACT.t_ambient + ACT.r_th * p_cu.mean(0)
    return p_bus, p_bus_no_regen, iq_rms, t_ss


def plot_envelope(joint, cols, d85, d100, path):
    w = np.linspace(-5, 5, 801)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 7.5))
    shades = {"full 54.6 V": "#1b1b1b", "nominal 46.8 V": "#555555", "low 40.0 V": "#a0a0a0"}
    for label, v in VOLTS.items():
        lo, hi = actuator.envelope(v, w)
        ax.plot(w, hi, color=shades[label], lw=1.6, label=f"envelope @ {label}")
        ax.plot(w, lo, color=shades[label], lw=1.6)
    for y, txt, ls in ((ACT.gear_repeated_peak, "gear repeated peak", "--"), (ACT.tau_cont_sw, "continuous", ":")):
        ax.axhline(y, color="#b03a2e", ls=ls, lw=1)
        ax.axhline(-y, color="#b03a2e", ls=ls, lw=1)
        ax.text(4.95, y + 2, txt, ha="right", va="bottom", color="#b03a2e", fontsize=8)
    for task, c in TASK_COLORS.items():
        for d, alpha, size, tag in ((d100, 0.18, 4, "100 kg"), (d85, 0.55, 5, "85 kg")):
            v = d[task]
            ax.scatter(v["qd"][::10, cols].ravel(), v["tau"][::10, cols].ravel(), s=size, color=c, alpha=alpha,
                       lw=0, label=f"{task.replace('_', ' ')} ({tag})")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-150, 150)
    ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
    ax.set_xlabel("joint speed [rad/s]  (+ = flexion)")
    ax.set_ylabel("joint torque [N m]  (+ = flexion)")
    ax.set_title(f"EXA-100 {joint}: output torque-speed envelope vs. demand")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8, frameon=False, ncol=3, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d85, d100 = demands(USER.mass), demands(USER.max_mass)
    stats = {"85kg": joint_stats(d85), "100kg": joint_stats(d100)}
    plot_envelope("hip", [0, 2], d85, d100, OUT / "torque_speed_hip.png")
    plot_envelope("knee", [1, 3], d85, d100, OUT / "torque_speed_knee.png")

    env = {}
    for label, v in VOLTS.items():
        w = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = actuator.envelope(v, w)
        env[label] = dict(speed_rad_s=w.tolist(), max_torque_Nm=np.round(hi, 1).tolist())
        ws = np.linspace(0, 6, 6001)
        _, h = actuator.envelope(v, ws)
        env[label]["no_load_speed_rad_s"] = round(float(ws[np.argmax(h <= 0)] if (h <= 0).any() else 6.0), 2)

    p_bus, p_bus_nr, iq_rms, t_ss = walking_power(d85)
    usable_wh = 0.9 * BATT.energy_wh
    speed = GAIT.step_length / GAIT.step_time
    runtime_h = usable_wh / p_bus
    peak_dem = {jt: max(stats["85kg"][t][jt]["peak_torque_Nm"] for t in TASK_COLORS) for jt in ("hip", "knee")}
    peak_dem100 = {jt: max(stats["100kg"][t][jt]["peak_torque_Nm"] for t in TASK_COLORS) for jt in ("hip", "knee")}
    peak_spd = {jt: max(stats["85kg"][t][jt]["peak_speed_rad_s"] for t in TASK_COLORS) for jt in ("hip", "knee")}
    margins = []
    for jt in ("hip", "knee"):
        margins += [
            dict(item=f"{jt} peak torque 85 kg vs software limit {ACT.tau_peak_sw:.0f} N m",
                 demand=peak_dem[jt], capacity=ACT.tau_peak_sw, margin=round(ACT.tau_peak_sw / peak_dem[jt] - 1, 3)),
            dict(item=f"{jt} peak torque 100 kg vs software limit",
                 demand=peak_dem100[jt], capacity=ACT.tau_peak_sw, margin=round(ACT.tau_peak_sw / peak_dem100[jt] - 1, 3)),
            dict(item=f"{jt} peak speed vs no-load speed @ 40 V",
                 demand=peak_spd[jt], capacity=env["low 40.0 V"]["no_load_speed_rad_s"],
                 margin=round(env["low 40.0 V"]["no_load_speed_rad_s"] / peak_spd[jt] - 1, 3)),
        ]
    for j, jn in enumerate(("hip_L", "knee_L")):
        margins.append(dict(item=f"{jn.split('_')[0]} RMS current (steady walking) vs continuous {ACT.i_cont} A",
                            demand=round(float(iq_rms[j]), 2), capacity=ACT.i_cont,
                            margin=round(ACT.i_cont / iq_rms[j] - 1, 3)))
    sts_stats = dynamics.inverse_dynamics(gait.make_sts(0.0), d85["walk"]["body"])
    report = dict(
        assumptions=dict(user_85kg_system_mass=round(d85["walk"]["body"].total, 1),
                         user_100kg_system_mass=round(d100["walk"]["body"].total, 1),
                         crutch_support_walk=GAIT.crutch_support_walk, crutch_support_sts=GAIT.crutch_support_sts,
                         sts_arm_push_extra=gait.STS_ARM_PUSH,
                         crutch_force_applied="vertical, through the HAT centre of mass",
                         gait=dict(step_length=GAIT.step_length, step_time=GAIT.step_time, speed_m_s=round(speed, 3))),
        joint_demands=stats,
        envelope=env,
        thermal_walking=dict(iq_rms_A=np.round(iq_rms, 2).tolist(), winding_steady_state_C=np.round(t_ss, 1).tolist(),
                             limit_C=ACT.t_max),
        battery=dict(pack=f"{BATT.series}S{BATT.parallel}P", energy_Wh=round(BATT.energy_wh, 1), usable_Wh=round(usable_wh, 1),
                     walking_bus_power_W=round(p_bus, 1), walking_bus_power_no_regen_W=round(p_bus_nr, 1),
                     continuous_walking_runtime_h=round(runtime_h, 2),
                     distance_per_charge_km=round(runtime_h * 3600 * speed / 1000, 2)),
        sts_knee_peak_without_arm_push_Nm=None,
        margins=margins,
    )
    push = gait.STS_ARM_PUSH
    gait.STS_ARM_PUSH = 0.0
    r0 = dynamics.inverse_dynamics(gait.make_sts(0.0), d85["walk"]["body"])
    gait.STS_ARM_PUSH = push
    report["sts_knee_peak_without_arm_push_Nm"] = round(float(np.abs(r0["tau"][:, 1]).max()), 1)
    report["sts_knee_peak_with_arm_push_Nm"] = round(float(np.abs(sts_stats["tau"][:, 1]).max()), 1)
    (OUT / "sizing.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: rep[k] for k in ("joint_demands", "thermal_walking", "battery", "margins",
                                          "sts_knee_peak_without_arm_push_Nm", "sts_knee_peak_with_arm_push_Nm")}, indent=1))
    print({k: v["no_load_speed_rad_s"] for k, v in rep["envelope"].items()})
