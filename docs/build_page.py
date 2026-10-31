"""
Build the project website from docs/page_template.html and the generated results.
Output: site/ (index.html plus every asset it references), ready for static hosting.

    python docs/build_page.py [--video video/out/Volition_simulation.mp4]
"""
import argparse
import datetime
import json
import math
import pathlib
import shutil
import subprocess
import sys

import imageio_ffmpeg
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()   # bundled; avoids a broken system ffmpeg
sys.path.insert(0, str(ROOT))
from params import ACT, EEG_CFG, GAIT, PRODUCT, STRUCT, USER  # noqa: E402

DOCS = ROOT / "docs"
SITE = ROOT / "site"
SIM = ROOT / "sim" / "out"
STILLS = ROOT / "video" / "out" / "stills"
REPO = "https://github.com/OjasMishra32/volition"


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------- envelopes
def envelope(v_bus, w_out):
    """Output-torque limits (hi, lo) against signed output speed: dq voltage circle,
    drive current limit and the software cap. Braking is looser than motoring."""
    lam = 2 * ACT.kt / (3 * ACT.pole_pairs)
    vm = v_bus * ACT.svpwm_utilization
    hi, lo = [], []
    for w in w_out:
        we = ACT.pole_pairs * ACT.ratio * abs(w)
        E, X, R = lam * we, we * ACT.l_phase, ACT.r_phase
        a = R * R + X * X
        disc = R * R * E * E - a * (E * E - vm * vm)
        i_mot = max(0.0, (-R * E + math.sqrt(disc)) / a) if disc >= 0 else 0.0
        i_brk = (R * E + math.sqrt(disc)) / a if disc >= 0 else 0.0
        t_mot = min(min(i_mot, ACT.i_peak) * ACT.kt * ACT.ratio * ACT.eff_forward, ACT.tau_peak_sw)
        t_brk = min(min(i_brk, ACT.i_peak) * ACT.kt * ACT.ratio / ACT.eff_backward, ACT.tau_peak_sw)
        hi.append(t_mot if w >= 0 else t_brk)
        lo.append(-t_brk if w >= 0 else -t_mot)
    return [round(v, 2) for v in hi], [round(v, 2) for v in lo]


# ------------------------------------------------------------- svg pieces
def montage_svg():
    u, cx, cy = 112.0, 190, 200
    sites = {"Cz": (0, 0), "C1": (-0.25, 0), "C2": (0.25, 0), "C3": (-0.5, 0), "C4": (0.5, 0), "C5": (-0.75, 0), "C6": (0.75, 0),
             "FCz": (0, 0.36), "FC3": (-0.46, 0.38), "FC4": (0.46, 0.38), "CPz": (0, -0.36), "CP3": (-0.46, -0.38), "CP4": (0.46, -0.38),
             "Pz": (0, -0.72), "AFz": (0, 1.08)}
    feet = {"Cz", "C1", "C2", "FCz", "CPz"}
    hands = {"C3", "C4", "C5", "C6", "FC3", "FC4", "CP3", "CP4"}
    out = []
    for name, (x, y) in sites.items():
        px, py = cx + x * u, cy - y * u
        cls = "fill-acc" if name in feet else "fill-sig" if name in hands else "fill-muted"
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{13 if name != "AFz" else 10}" class="{cls}"/>')
        out.append(f'<text x="{px:.1f}" y="{py + 3.8:.1f}" text-anchor="middle" font-size="9.5" font-weight="600" style="fill:#fff">{name}</text>')
    out += ['<text x="28" y="204" text-anchor="middle" font-size="10.5" class="t-muted">A1</text>',
            '<text x="352" y="204" text-anchor="middle" font-size="10.5" class="t-muted">A2</text>']
    return "\n            ".join(out)


def bridges_svg():
    out, ys = [], [150, 168, 186]
    for k in range(3):
        x = 270 + 110 * k
        out += [f'<path d="M{x} 40 V74" class="wire"/>',
                f'<rect x="{x - 15}" y="74" width="30" height="48" rx="6" class="box"/><text x="{x}" y="103" font-size="11.5" text-anchor="middle">Q{2 * k + 1}</text>',
                f'<path d="M{x} 122 V210" class="wire"/>',
                f'<circle cx="{x}" cy="{ys[k]}" r="3" class="fill-ink"/>',
                f'<path d="M{x} {ys[k]} H724" class="wire"/>',
                f'<rect x="{x - 15}" y="210" width="30" height="48" rx="6" class="box"/><text x="{x}" y="239" font-size="11.5" text-anchor="middle">Q{2 * k + 2}</text>',
                f'<path d="M{x} 258 V280" class="wire"/>',
                f'<rect x="{x - 9}" y="280" width="18" height="24" rx="3" class="box-soft"/>',
                f'<path d="M{x} 304 V330" class="wire"/>']
    for k, ph in enumerate("UVW"):
        out.append(f'<text x="600" y="{ys[k] - 4}" font-size="11" class="t-muted">{ph}</text>')
    out += ['<text x="290" y="320" font-size="11" class="t-muted">1 mΩ shunts → current-sense amps → ADC at PWM centre</text>',
            '<text x="246" y="64" font-size="11" class="t-muted" text-anchor="end">80 V FETs</text>']
    return "\n          ".join(out)


def chain_svg():
    blocks = [("Slew", "±2500 N·m/s"), ("Peak", "±100 N·m"), ("Current", "22 A → ±148"),
              ("Voltage", "f(ω, Vbus, R, L)"), ("Thermal", "100 → 120 °C")]
    out, x = [], 376
    for k, (a, b) in enumerate(blocks):
        out.append(f'<rect x="{x}" y="100" width="112" height="64" rx="14" class="box-acc"/>')
        out.append(f'<text x="{x + 56}" y="126" font-size="13" font-weight="600" text-anchor="middle">{a}</text>')
        out.append(f'<text x="{x + 56}" y="146" font-size="11" class="t-muted" text-anchor="middle">{b}</text>')
        if k < len(blocks) - 1:
            out.append(f'<path d="M{x + 112} 132 H{x + 124}" class="wire-acc" marker-end="url(#ar6a)"/>')
        x += 126
    return "\n          ".join(out)


# ------------------------------------------------------------- anti-windup
def antiwindup_series():
    """Rerun the simulation's own low-battery sit-to-stand study for the charts."""
    from control import simulate
    naive, aw = simulate.sts_study(False), simulate.sts_study(True)
    j = 3
    t = aw["t"]
    step = max(1, len(t) // 500)
    e_n = np.degrees(naive["q_ref"][:, j] - naive["q"][:, j])
    e_a = np.degrees(aw["q_ref"][:, j] - aw["q"][:, j])
    sat = (aw["reason"][:, j] >= 1) & (aw["reason"][:, j] <= 4)
    spans, start = [], None
    for k, s in enumerate(sat):
        if s and start is None:
            start = t[k]
        if not s and start is not None:
            spans.append([round(float(start), 3), round(float(t[k]), 3)])
            start = None
    if start is not None:
        spans.append([round(float(start), 3), round(float(t[-1]), 3)])
    em = max(np.abs(e_n).max(), np.abs(e_a).max()) * 1.15
    im = max(np.abs(naive["integ"][:, j]).max(), np.abs(aw["integ"][:, j]).max()) * 1.15
    nice = lambda v, s: math.ceil(v / s) * s  # noqa: E731
    es, is_ = nice(em, 0.5), nice(im, 20)
    r = lambda a: [round(float(v), 3) for v in a[::step]]  # noqa: E731
    T = float(t[-1])
    return {"t": r(t), "e_naive": r(e_n), "e_aw": r(e_a), "i_naive": r(naive["integ"][:, j]), "i_aw": r(aw["integ"][:, j]),
            "spans": spans, "xt": [round(v, 1) for v in np.arange(0, T + 0.01, 1.0)],
            "e_y": [-es, es], "e_yt": [round(v, 2) for v in np.linspace(-es, es, 5)],
            "i_y": [-is_ * 0.25, is_], "i_yt": [round(v) for v in np.linspace(0, is_, 5)]}


# ------------------------------------------------------------------ tables
BOM = [
    ("Frameless BLDC motor, ~85 mm", "0.09 N·m/A, 10 pole pairs, 48 V", 4, 180),
    ("Strain-wave gear set", "size 25, 100:1", 4, 650),
    ("Crossed-roller output bearing", "Ø96 × 10 mm", 4, 90),
    ("Torque-sensor flexure and gauges", "7075 flexure, full bridge, ±150 N·m", 4, 120),
    ("Absolute output encoder", "19-bit ring, BiSS-C", 4, 140),
    ("Motor encoder IC and magnet", "14-bit on-axis", 4, 8),
    ("Power-off brake", "spring-applied, 1.5 N·m, 12 V", 4, 110),
    ("Field-oriented servo drive", "48 V, 25 A peak, CAN-FD, two-channel STO", 4, 160),
    ("Safety controller board", "lockstep MCU, CAN-FD, IMU, I/O", 1, 250),
    ("Application processor module", "quad-core SoM, 4 GB", 1, 120),
    ("EEG front-end board", "2× ADS1299, isolation, isolated DC/DC", 1, 180),
    ("EEG cap and electrodes", "16 channels, gel Ag/AgCl", 1, 600),
    ("Battery pack with BMS", "13S2P 21700, 468 Wh, hot-swap", 2, 320),
    ("IMU modules", "6-axis, SPI", 5, 15),
    ("Pressure insoles", "4-zone FSR, pair", 1, 120),
    ("Smart forearm crutches", "buttons, load cell, IMU, pair", 1, 260),
    ("Structure", "carbon tube, 7075 inserts, pelvis band, uprights", 1, 900),
    ("Cuffs, liners, straps, AFOs", "thermoformed shells, spacer-fabric liners", 1, 350),
    ("Stop chain", "3 stop buttons, safety relay, caregiver remote", 1, 180),
    ("Harness, connectors, fasteners", "", 1, 250),
]

FILES = [
    ("params.py", "One parameter file: user, structure, actuator, battery, EEG, control, gait."),
    ("cad/build_cad.py", "Parametric CadQuery model of every part."),
    ("cad/out/Volition_standing.step", "Full coloured assembly, standing pose."),
    ("cad/out/EXA-100_actuator.step", "The actuator with its internals."),
    ("control/gait.py", "Cartesian gait and sit-stand planner with inverse kinematics."),
    ("control/dynamics.py", "Planar Newton-Euler inverse dynamics with crutch and chair support."),
    ("control/actuator.py", "Motor, gear, drive, battery and bus model; the five-limit chain."),
    ("control/joint_controller.py", "Feed-forward + PID with anti-windup."),
    ("control/bci.py", "Synthetic EEG, filtering, artefact gating, features, LDA, evidence."),
    ("control/supervisor.py", "State machine, intent gate, safety monitors."),
    ("control/simulate.py", "1 kHz closed-loop scenario and anti-windup study."),
    ("design/sizing.py", "Actuator sizing, torque-speed envelopes, thermal check and runtime."),
    ("video/render.py", "Off-screen renderer and overlay for the film."),
    ("docs/build_page.py", "Builds this website from the template and the simulation results."),
]


def finding_rows(meta, sizing, aw):
    sts = sizing.get("sts_knee_peak_without_arm_push_Nm", 116.3)
    with_push = sizing.get("sts_knee_peak_with_arm_push_Nm", 101.8)
    m100 = next((m for m in sizing.get("margins", []) if m["item"].startswith("knee peak torque 100 kg")), {"demand": 116.2})
    ankle = sizing.get("joint_demands", {}).get("85kg", {}).get("walk", {}).get("peak_ankle_moment_Nm", 174.0)
    lat = meta.get("bci", {}).get("detection_latency_s", [3.3, 4.7])
    runtime = sizing.get("battery", {}).get("continuous_walking_runtime_h", 7.6)
    vmin = aw.get("v_bus_range", [39.0])[0]
    data = [
        ("Knee torque at sit-to-stand", f"{with_push:.1f} N·m",
         f"An 85 kg user needs {with_push:.1f} N·m at the knee with an extra 15% push on the crutches, and {sts:.0f} N·m without it. "
         f"A 100 kg user needs {m100['demand']:.0f} N·m.",
         "Use a size-32 gear stage at the knees, or limit users to 90 kg."),
        ("Passive ankle", f"{ankle:.0f} N·m",
         "Peak stance ankle moment while walking, far more than the 120 N·m/rad AFO spring provides. Dorsiflexion reaches about 24°, past the 20° stop.",
         "Fit a stiffer AFO with an adjustable stop, or add powered ankles."),
        ("Standing on a low battery", f"{vmin:.1f} V",
         "With the battery at 41.6 V open-circuit, the bus drops below the 40 V forced-sit threshold during seat-off.",
         "Refuse to stand below about 44 V open-circuit."),
        ("BCI detection time", f"{min(lat):.1f} to {max(lat):.1f} s",
         "Time from the start of imagery to a detection. That is workable for discrete commands and too slow for continuous control.",
         "Calibrate per user, adapt thresholds over time and keep button-only control available."),
        ("Battery runtime", f"{runtime:.1f} h",
         "Continuous-walking estimate. It leaves out iron losses, standby electronics and sit-stand cycles.",
         "Plan on 3 to 4 h of mixed use until it is measured."),
        ("Trunk lean at sit-to-stand", "34°",
         "The hip range limits trunk lean to about 34° instead of the planned 38°, which is why the extra crutch push is needed.",
         "Extend hip flexion range to 120°."),
    ]
    return rows([(a, (b,), c, d) for a, b, c, d in data])


def rows(data):
    out = []
    for r in data:
        cells = "".join(f'<td class="num">{c[0]}</td>' if isinstance(c, tuple) else f"<td>{c}</td>" for c in r)
        out.append(f"<tr>{cells}</tr>")
    return "\n        ".join(out)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "video" / "out" / f"{PRODUCT}_simulation.mp4"))
    ap.add_argument("--web-crf", type=int, default=25)
    a = ap.parse_args()
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    meta = load_json(SIM / "run_meta.json")
    aw_json = load_json(SIM / "antiwindup.json")
    sizing = load_json(ROOT / "design" / "out" / "sizing.json")
    render_meta = load_json(ROOT / "video" / "out" / "render_meta.json")
    z = np.load(SIM / "run.npz", allow_pickle=True)
    t = z["t"]
    names = [str(s) for s in z["state_names"]]
    state = z["state"].astype(int)

    mass = sum([STRUCT.m_pelvis_frame, STRUCT.m_backpack, 4 * ACT.mass, 2 * STRUCT.m_thigh_link,
                2 * STRUCT.m_shank_link, 2 * STRUCT.m_foot, 0.35])
    dem = sizing["joint_demands"]["85kg"]
    sizing_rows = []
    for task, label in (("walk", "walking"), ("sit_to_stand", "sit to stand"), ("stand_to_sit", "stand to sit")):
        for jn in ("hip", "knee"):
            d = dem[task][jn]
            margin = 1 - d["peak_torque_Nm"] / ACT.tau_peak_sw
            sizing_rows.append((f"{jn.title()}, {label}", (f"{d['peak_torque_Nm']:.1f}",), (f"{d['rms_torque_Nm']:.1f}",),
                                (f"{d['peak_speed_rad_s']:.2f}",), (f"{d['peak_power_W']:.0f}",), (f"{margin * 100:+.0f}%",)))

    # torque-speed envelopes and simulated operating points (every 10 ms)
    q, tau = z["q"], z["tau_out"]
    dq = np.gradient(q, t, axis=0)
    walk = np.isin(state, [names.index(s) for s in ("WALKING", "STOPPING")])
    sts = np.isin(state, [names.index(s) for s in ("STANDING_UP", "SITTING_DOWN")])
    w = np.linspace(-6.0, 6.0, 241)
    env = {"w": [round(float(v), 3) for v in w], "wmax": 6.0, "xt": [-6, -4, -2, 0, 2, 4, 6]}
    for key, v in (("v546", 54.6), ("v468", 46.8), ("v400", 40.0)):
        hi, lo = envelope(v, w)
        env[key] = {"hi": hi, "lo": lo}
    step = max(1, int(round(0.01 / np.median(np.diff(t)))))
    points = {}
    for jn, js in (("hip", (0, 2)), ("knee", (1, 3))):
        points[jn] = {}
        for task, mask in (("walk", walk), ("sts", sts)):
            idx = np.nonzero(mask)[0][::step]
            points[jn][task] = np.round(np.concatenate([np.c_[dq[idx, j], tau[idx, j]] for j in js]), 2).tolist()

    results = {"envelope": env, "points": points, "antiwindup": antiwindup_series(),
               "geometry": {"thigh_len": USER.thigh_len, "shank_len": USER.shank_len, "hjc_y": USER.hip_half_width}}

    tp = np.arange(0, t[-1], 0.04)
    cols = [z["pelvis_xz"][:, 0], z["pelvis_xz"][:, 1], z["trunk_pitch"], z["phi_thigh_L"], z["phi_shank_L"], z["psi_foot_L"],
            z["phi_thigh_R"], z["phi_shank_R"], z["psi_foot_R"]]
    frames = np.stack([np.interp(tp, t, c) for c in cols], axis=1)
    st_idx = state[np.clip(np.searchsorted(t, tp), 0, len(t) - 1)]
    pose = {"dt": 0.04, "states": names, "frames": [[round(float(v), 4) for v in f] + [int(s)] for f, s in zip(frames, st_idx)]}

    bci = meta.get("bci", {})
    fa = bci.get("false_activation", {})
    rest_min = sum(v.get("minutes", 0) for v in fa.values())
    fa_events = sum(v.get("events", 0) for v in fa.values())
    lat = bci.get("detection_latency_s", [0])
    nw, aw = aw_json.get("naive_pid", {}), aw_json.get("back_calculation", {})
    vr = aw_json.get("v_bus_range", [0, 0])
    aw_caption = (f"Anti-windup test: sit-to-stand on a nearly empty battery ({vr[0]:.1f} to {vr[1]:.1f} V on the bus), where the knee limit "
                  f"drops as low as {aw_json.get('torque_limit_min_Nm', 0):.0f} N·m. Plain PID winds its integrator up to "
                  f"{nw.get('integrator_peak_Nm', 0):.0f} N·m and overshoots by {nw.get('overshoot_deg', 0):.2f}° when the limit releases. "
                  f"With anti-windup the integrator peaks at {aw.get('integrator_peak_Nm', 0):.0f} N·m and the overshoot is {aw.get('overshoot_deg', 0):.2f}°.")
    subs = {
        "date": datetime.date.today().isoformat(),
        "exo_mass": f"{mass:.1f}",
        "video_len": f"{render_meta.get('duration_s', t[-1]):.0f}",
        "n_steps": str(meta.get("steps_walked", 8)),
        "walk_speed": f"{sizing['assumptions']['gait']['speed_m_s']:.2f}",
        "step_len": f"{GAIT.step_length:.2f}", "step_time": f"{GAIT.step_time:.1f}",
        "bci_latency": f"{min(lat):.1f} to {max(lat):.1f}",
        "bci_acc": f"{bci.get('calibration', {}).get('heldout_window_accuracy', 0) * 100:.0f}%",
        "bci_fa": f"{fa_events / max(rest_min, 1e-9):.1f}", "bci_fa_total": str(fa_events), "bci_rest_min": f"{rest_min:.0f}",
        "bci_reject": f"{bci.get('windows_rejected_frac', 0) * 100:.0f}%",
        "confirm_timeout": f"{EEG_CFG.confirm_timeout_s:.0f}", "ev_tau": f"{EEG_CFG.evidence_tau_s:.1f}",
        "ev_thr": f"{EEG_CFG.evidence_threshold:.2f}", "dwell": f"{EEG_CFG.min_dwell_s:.2f}", "refractory": f"{EEG_CFG.refractory_s:.0f}",
        "clearance": f"{GAIT.swing_clearance * 100:.0f}", "stance_knee": f"{math.degrees(GAIT.stance_knee_target):.0f}",
        "ds_pct": f"{GAIT.double_support_frac * 100:.0f}",
        "walk_power": f"{sizing['battery']['walking_bus_power_W']:.0f}",
        "pk_knee_sts": f"{dem['sit_to_stand']['knee']['peak_torque_Nm']:.0f}",
        "pk_hip_sts": f"{dem['sit_to_stand']['hip']['peak_torque_Nm']:.0f}",
        "aw_caption": aw_caption,
        "montage": montage_svg(), "bridges": bridges_svg(), "chain_blocks": chain_svg(),
        "sizing_rows": rows(sizing_rows),
        "finding_rows": finding_rows(meta, sizing, aw_json),
        "bom_rows": rows([(n, s, (q_,), (f"${c:,}",), (f"${q_ * c:,}",)) for n, s, q_, c in BOM]),
        "bom_total": f"${sum(q_ * c for _, _, q_, c in BOM):,}",
        "file_rows": "\n      ".join(f'<div><a href="{REPO}/blob/main/{f}"><code>{f}</code></a>{d}</div>' for f, d in FILES),
        "repo": REPO,
        "RESULTS_JSON": json.dumps(results, separators=(",", ":")),
        "POSE_JSON": json.dumps(pose, separators=(",", ":")),
    }
    html = (DOCS / "page_template.html").read_text()
    for k, v in subs.items():
        html = html.replace(f"@@{k}@@", v)
    left = sorted({p for p in html.split("@@")[1::2] if p.isidentifier()})
    if left:
        print("WARNING unreplaced placeholders:", left)
    (SITE / "index.html").write_text(html)

    shutil.copy(ROOT / "cad" / "out" / "volition_web.glb", SITE / "volition_web.glb")
    for f in STILLS.glob("*"):
        if f.suffix in (".png", ".jpg"):
            shutil.copy(f, SITE / f.name)
    vid = pathlib.Path(a.video)
    if vid.exists():
        web = SITE / "volition_sim.mp4"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(vid), "-c:v", "libx264", "-crf", str(a.web_crf), "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(web)], check=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", "21", "-i", str(vid), "-frames:v", "1", "-q:v", "3",
                        str(SITE / "volition_poster.jpg")], check=True)
        print(f"web video {web.stat().st_size / 1e6:.1f} MB")
    print("site ->", SITE, f"index {(SITE / 'index.html').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
