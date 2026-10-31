"""
Render the closed-loop simulation (sim/out/run.npz) to a 1080p H.264 film.

    python video/render.py [--fps 60] [--workers 6] [--still t ...]

Each worker builds its own off-screen scene and overlay, renders a contiguous
chunk of frames and encodes it; chunks are joined losslessly at the end.
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()   # bundled; avoids a broken system ffmpeg
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "video"))
from params import PRODUCT  # noqa: E402

RUN = ROOT / "sim" / "out" / "run.npz"
OUT = ROOT / "video" / "out"
SIZE = (1920, 1080)
JOINTS = ("hip_L", "knee_L", "hip_R", "knee_R")
TAIL = 3.2                     # seconds held after the simulation ends (end card)
TITLE_END = 3.0
STATE_NAMES = {"SITTING": "Seated", "STANDING_UP": "Standing up", "STANDING": "Standing", "WALKING": "Walking",
               "STOPPING": "Stopping", "SITTING_DOWN": "Sitting down", "FAULT_HOLD": "Holding"}
REASON_TXT = {1: "Peak limit", 2: "Current", 3: "Voltage", 4: "Thermal", 5: "Slew"}


# ------------------------------------------------------------------ data access
class Run:
    def __init__(self, path=RUN):
        z = np.load(path, allow_pickle=True)
        self.z = {k: z[k] for k in z.files}
        self.t = self.z["t"]
        self.state_names = [str(s) for s in self.z["state_names"]]
        self.T = float(self.t[-1])

    def at(self, key, t):
        a = self.z[key]
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        w = float(np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0, 1))
        return a[i] * (1 - w) + a[i + 1] * w

    def hold(self, key, t, tkey="t"):
        tt = self.z[tkey]
        i = int(np.clip(np.searchsorted(tt, t, side="right") - 1, 0, len(tt) - 1))
        return self.z[key][i]

    def window(self, key, t, span, tkey="t", step=1):
        tt = self.z[tkey]
        i0, i1 = np.searchsorted(tt, [t - span, t])
        return tt[i0:i1 + 1:step], self.z[key][i0:i1 + 1:step]

    def first_time(self, state):
        idx = np.nonzero(self.z["state"] == self.state_names.index(state))[0]
        return float(self.t[idx[0]]) if len(idx) else None

    def last_time(self, state):
        idx = np.nonzero(self.z["state"] == self.state_names.index(state))[0]
        return float(self.t[idx[-1]]) if len(idx) else None


def smooth(x, n):
    if n < 1:
        return x
    k = np.hanning(2 * n + 1)
    k /= k.sum()
    return np.convolve(np.r_[np.full(n, x[0]), x, np.full(n, x[-1])], k, mode="valid")


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------- captions
def caption_track(run):
    """Short keynote captions from the simulation's event log."""
    ev = sorted(zip(run.z["event_t"].astype(float), [str(s) for s in run.z["event_text"]]))
    seen = {"feet": 0, "hands": 0}
    caps = []
    for t, s in ev:
        lo = s.lower()
        text = None
        if "imagines moving feet" in lo:
            text = ("The user imagines pressing both feet down.", "Foot imagery again, this time to walk.")[min(seen["feet"], 1)]
            seen["feet"] += 1
        elif "imagines moving both hands" in lo:
            text = ("To stop, the user imagines squeezing both hands.", "Hand imagery again, this time to sit.")[min(seen["hands"], 1)]
            seen["hands"] += 1
        elif "detected" in lo:
            text = "The decoder detects the intent and waits for the button."
        elif "confirmed" in lo:
            text = {"stand": "Button pressed. Standing up.", "walk": "Button pressed. Walking.",
                    "stop": "Button pressed. Stopping after this step.", "sit": "Button pressed. Sitting down."}.get(lo.rsplit(">", 1)[-1].strip(), "Button pressed.")
        elif "torque limit" in lo:
            text = "At seat-off the knees reach their 100 N·m limit."
        elif "electrode pop" in lo:
            text = "Electrode C4 pops, so those EEG windows are rejected."
        elif "signal clean" in lo:
            text = "The signal is clean again and decoding resumes."
        elif lo.startswith("stopped"):
            text = "Stopped."
        elif lo == "standing":
            text = "Standing."
        elif lo == "seated":
            text = "Seated."
        if text and not (caps and abs(caps[-1][0] - t) < 0.05 and caps[-1][1] == text):
            if caps and abs(caps[-1][0] - t) < 0.05:
                caps[-1] = (t, text)
            else:
                caps.append((t, text))
    out = []
    for k, (t, text) in enumerate(caps):
        nxt = caps[k + 1][0] if k + 1 < len(caps) else run.T + 0.6
        out.append((t, min(nxt, t + 4.2), text))
    return out


def caption_at(caps, t):
    for a, b, text in caps:
        if a <= t < b:
            return text, float(np.clip(min((t - a) / 0.3, (b - t) / 0.35), 0, 1))
    return "", 0.0


# ------------------------------------------------------------------- camera
def camera_track(run, frame_t):
    from scipy.interpolate import PchipInterpolator
    T = run.T
    su, st, wk = run.first_time("STANDING_UP"), run.first_time("STANDING"), run.first_time("WALKING")
    sp, sd = run.first_time("STOPPING"), run.first_time("SITTING_DOWN")
    stood2 = run.last_time("STOPPING") or sp
    walk_mid = (wk + sp) / 2
    keys = [(0.0, 24, 8, 3.7, 0.80), (su - 0.4, 34, 8, 4.1, 0.82), (st + 0.3, 46, 8, 4.7, 0.88), (wk - 0.6, 58, 8, 4.9, 0.88),
            (wk + 1.6, 72, 7, 5.3, 0.88), (walk_mid, 90, 7, 5.4, 0.88), (sp, 102, 8, 5.2, 0.88), (stood2 + 1.2, 74, 8, 4.9, 0.88),
            (sd, 48, 8, 4.4, 0.84), (T, 34, 8, 4.0, 0.80), (T + TAIL, 28, 9, 3.7, 0.80)]
    kt = np.array([k[0] for k in keys])
    f = {i: PchipInterpolator(kt, [k[i] for k in keys])(frame_t) for i in range(1, 5)}
    px = np.array([run.at("pelvis_xz", min(t, run.T))[0] for t in frame_t])
    fps = 1 / np.median(np.diff(frame_t))
    fxs = smooth(px, int(fps * 0.9)) + 0.08
    cams = []
    for a, e, d, fz, x in zip(np.radians(f[1]), np.radians(f[2]), f[3], f[4], fxs):
        focal = np.array([x, 0.0, fz])
        pos = focal + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        right = np.cross(focal - pos, [0, 0, 1])
        right /= np.linalg.norm(right)
        shift = right * 0.093 * d        # keeps the user centred in the area left of the glass card
        cams.append((pos + shift, focal + shift))
    return cams


def chairs_from_run(run):
    xs = [float(x) for x in run.z["chairs_x"]]
    seat_top = float(run.z["seat_height"])
    hip0 = float(run.z["pelvis_xz"][0, 0])
    front_off = 0.30 + (hip0 - xs[0])      # seat front edge sits ~0.30 m ahead of the seated hip
    return [(x + front_off, seat_top) for x in xs]


def chair2_offset(run, t):
    """Second chair slides in from the user's right after they stop."""
    t0 = (run.last_time("STOPPING") or run.T) + 0.2
    k = ease((t - t0) / 1.8)
    return (0.0, -1.6 * (1 - k), 0.0), t > t0 - 0.05


# ------------------------------------------------------------------- overlay
def hud_state(run, t, caps):
    from hud import CYAN, ORANGE, RED, SECOND
    ts = min(t, run.T)
    d = {"t": ts}
    sname = run.state_names[int(run.hold("state", ts))]
    d["state"] = STATE_NAMES.get(sname, sname.title())
    v = float(run.at("pelvis_xz", ts)[0] - run.at("pelvis_xz", max(ts - 1.0, 0))[0])
    if sname == "WALKING":
        n = int(np.sum(run.z["step_t"] <= ts))
        d["state_sub"] = f"Step {max(n, 1)} of {len(run.z['step_t'])}  ·  {v:.2f} m/s"
    else:
        d["state_sub"] = {"SITTING": "EEG decoding · 16 Hz", "STANDING_UP": "Sit-to-stand · 3.0 s",
                          "STANDING": "Balanced on crutches", "STOPPING": "Closing step",
                          "SITTING_DOWN": "Stand-to-sit · recovering energy", "FAULT_HOLD": "Posture held"}.get(sname, "")

    et, eeg = run.window("eeg_display", ts, 4.0, tkey="eeg_t", step=2)
    d["eeg_t"], d["eeg"] = et, (np.clip(eeg / 45.0, -0.46, 0.46) if len(eeg) else np.zeros((0, 3)))
    ct, cf = run.window("eeg_clip", ts, 4.0, tkey="eeg_t", step=2)
    d["clip_span"] = (float(ct[cf][0]), float(ct[cf][-1]) + 0.05) if cf.any() else None
    classes = [str(c) for c in run.z["bci_classes"]]
    prob, evid = run.hold("bci_prob", ts, tkey="bci_t"), run.hold("bci_evidence", ts, tkey="bci_t")
    valid = bool(run.hold("bci_valid", ts, tkey="bci_t"))
    thr = float(run.z["bci_threshold"])
    d["prob"] = {c: float(prob[k]) for k, c in enumerate(classes)}
    d["evidence"] = {c: float(evid[k]) for k, c in enumerate(classes)}
    if not valid:
        d["bci_line"] = ("Electrode pop · windows rejected", RED)
    elif d["evidence"]["FEET_MI"] >= thr:
        d["bci_line"] = ("Feet intent · waiting for crutch button", ORANGE)
    elif d["evidence"]["HANDS_MI"] >= thr:
        d["bci_line"] = ("Hands intent · waiting for crutch button", CYAN)
    else:
        d["bci_line"] = ("Decoding · 16 updates per second", SECOND)

    tt, out = run.window("tau_out", ts, 5.0, step=2)
    _, hi = run.window("tau_lim_pos", ts, 5.0, step=2)
    _, lo = run.window("tau_lim_neg", ts, 5.0, step=2)
    _, rs = run.window("sat_reason", ts, 5.0, step=2)
    d["tq_t"] = tt
    d["out"] = {j: out[:, k] for k, j in enumerate(JOINTS)}
    d["lim_hi"] = {j: np.minimum(hi[:, k], 125) for k, j in enumerate(JOINTS)}
    d["lim_lo"] = {j: np.maximum(lo[:, k], -125) for k, j in enumerate(JOINTS)}
    d["satm"] = {j: (rs[:, k] >= 1) & (rs[:, k] <= 4) for k, j in enumerate(JOINTS)}   # slew smoothing is not a limit
    on, rn = run.at("tau_out", ts), run.hold("sat_reason", ts)
    d["out_now"] = {j: float(on[k]) for k, j in enumerate(JOINTS)}
    d["reason_now"] = {j: int(rn[k]) if 1 <= rn[k] <= 4 else 0 for k, j in enumerate(JOINTS)}
    d["reason_txt"] = {j: REASON_TXT.get(int(rn[k]), "") for k, j in enumerate(JOINTS)}

    d["vbus"], d["soc"] = float(run.at("v_bus", ts)), float(run.at("soc", ts))
    pel = float(np.sum(run.at("p_elec", ts)))
    tw = float(np.max(run.at("T_wind", ts)))
    d["pwr_line"] = f"{'Recovering' if pel < -3 else 'Drawing'} {abs(pel):.0f} W  ·  motors {tw:.0f} °C"

    d["caption"] = caption_at(caps, t) if TITLE_END <= t <= run.T + 0.6 else ("", 0.0)
    d["ui_alpha"] = float(np.clip((t - (TITLE_END - 0.4)) / 0.9, 0, 1)) * float(np.clip((run.T + 1.2 - t) / 0.8, 0, 1))
    if t < TITLE_END:
        d["title"], d["title2"] = PRODUCT, "A brain-controlled exoskeleton, simulated"
        d["title_alpha"] = float(np.clip(min(t / 0.5 + 0.2, (TITLE_END - t) / 0.6), 0, 1))
    elif t > run.T + 0.4:
        d["title"], d["title2"] = PRODUCT, "Concept design. Simulated physics and synthetic EEG."
        d["title_alpha"] = float(np.clip((t - run.T - 0.4) / 0.8, 0, 1))
    return d


def glass_alpha(w, h, r):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    dx = np.maximum(np.maximum(r - xx, xx - (w - r)), 0)
    dy = np.maximum(np.maximum(r - yy, yy - (h - r)), 0)
    return np.clip(r - np.hypot(dx, dy) + 0.5, 0, 1)


def apply_glass(img, rect, alpha, edge):
    x, y, w, h = rect
    reg = img[y:y + h, x:x + w]
    small = ndimage.gaussian_filter(reg[::4, ::4], (5, 5, 0))
    blur = ndimage.zoom(small, (h / small.shape[0], w / small.shape[1], 1), order=1)[:h, :w]
    glass = blur * 0.38 + np.array([26.0, 26.0, 30.0]) * 0.62
    a = alpha[..., None]
    reg[:] = reg * (1 - a) + glass * a + edge[..., None] * 22.0


def vignette(h, w, strength=0.28):
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt(((x - w * 0.38) / (w * 0.7)) ** 2 + ((y - h * 0.5) / (h * 0.8)) ** 2)
    v = 1 - strength * np.clip(r, 0, 1.2) ** 2.4
    v *= 1 - 0.5 * np.clip((y - 860) / 220, 0, 1) ** 1.5 * np.clip((1450 - x) / 300, 0, 1)   # caption scrim
    return v[..., None].astype(np.float32)


def pose_state(run, ts, util=None, sat=None, mi=None):
    """Everything Scene.pose needs at simulation time ts."""
    st = {"pelvis_xz": run.at("pelvis_xz", ts), "trunk_pitch": float(run.at("trunk_pitch", ts)),
          "util": util or {}, "sat": sat or {}, "mi": mi or {}}
    for s in ("L", "R"):
        for k in ("phi_thigh", "phi_shank", "psi_foot"):
            st[f"{k}_{s}"] = float(run.at(f"{k}_{s}", ts))
        st[f"hand_xyz_{s}"] = run.at(f"hand_xyz_{s}", ts)
        st[f"crutch_tip_xyz_{s}"] = run.at(f"crutch_tip_xyz_{s}", ts)
    return st


# ------------------------------------------------------------------- worker
def worker(args):
    idx0, idx1, fps, seg_path = args
    import imageio_ffmpeg
    from hud import CARD, HUD
    from scene import Scene

    run = Run()
    frame_t = np.arange(0, run.T + TAIL, 1 / fps)
    cams = camera_track(run, frame_t)
    caps = caption_track(run)
    scene = Scene(window_size=SIZE, chairs=chairs_from_run(run))
    hud = HUD(thr=float(run.z["bci_threshold"]), name=PRODUCT)
    classes = [str(c) for c in run.z["bci_classes"]]
    vig = vignette(SIZE[1], SIZE[0])
    x, y, w, h = CARD
    alpha = glass_alpha(w, h, 30)
    edge = np.clip(alpha - ndimage.binary_erosion(alpha > 0.5, iterations=1), 0, 1)
    mi = {"FEET": 0.0, "HANDS": 0.0}

    writer = None
    if seg_path:
        writer = imageio_ffmpeg.write_frames(str(seg_path), SIZE, fps=fps, codec="libx264", macro_block_size=8, pix_fmt_out="yuv420p",
                                             quality=None, output_params=["-crf", "14", "-preset", "slow", "-profile:v", "high", "-tune", "film"])
        writer.send(None)
    frames = []
    for i in range(idx0, idx1):
        t = float(frame_t[i])
        ts = min(t, run.T)
        tau, hi, lo = run.at("tau_out", ts), run.at("tau_lim_pos", ts), run.at("tau_lim_neg", ts)
        rs = run.hold("sat_reason", ts)
        util = {j: float(abs(tau[k]) / max(abs(hi[k] if tau[k] >= 0 else lo[k]), 1e-3)) for k, j in enumerate(JOINTS)}
        sat = {j: bool(rs[k] in (1, 2, 3, 4)) for k, j in enumerate(JOINTS)}
        prob = run.hold("bci_prob", ts, tkey="bci_t")
        valid = bool(run.hold("bci_valid", ts, tkey="bci_t"))
        for key, cls in (("FEET", "FEET_MI"), ("HANDS", "HANDS_MI")):
            target = max(0.0, (float(prob[classes.index(cls)]) - 0.4) / 0.6) if valid else 0.0
            mi[key] += (target - mi[key]) * (1 - math.exp(-6.0 / fps))
        scene.pose(pose_state(run, ts, util, sat, mi))
        if len(scene.chairs) > 1:
            off, vis = chair2_offset(run, t)
            scene.set_chair(1, off, vis)
        scene.camera(*cams[i], view_angle=24)
        img = scene.image() * vig
        hd = hud_state(run, t, caps)
        if hd.get("title_alpha", 0) > 0:
            img *= 1 - 0.55 * hd["title_alpha"]
        ui = hd["ui_alpha"]
        if ui > 0:
            apply_glass(img, CARD, alpha * ui, edge * ui)
        over = hud.draw(hd).astype(np.float32)
        over[: 220, : 900, 3] *= ui                      # state block
        over[:, CARD[0] - 4:, 3] *= ui                   # glass card
        a = over[..., 3:4] / 255.0
        img = np.clip(img * (1 - a) + over[..., :3] * a, 0, 255).astype(np.uint8)
        if writer:
            writer.send(np.ascontiguousarray(img))
        else:
            frames.append(img)
    if writer:
        writer.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--still", type=float, nargs="*", help="render stills at these times (s) to video/out/stills")
    ap.add_argument("--name", default=f"{PRODUCT}_simulation.mp4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run = Run()
    n = len(np.arange(0, run.T + TAIL, 1 / a.fps))

    if a.still:
        from PIL import Image
        (OUT / "frames").mkdir(exist_ok=True)
        for ts in a.still:
            i = min(int(round(ts * a.fps)), n - 1)
            img = worker((i, i + 1, a.fps, None))[0]
            Image.fromarray(img).save(OUT / "frames" / f"t{ts:05.1f}.png")
            print("still", ts)
        return

    t0 = time.time()
    bounds = np.linspace(0, n, a.workers + 1).astype(int)
    segs = [OUT / f"_seg{k:02d}.mp4" for k in range(a.workers)]
    jobs = [(int(bounds[k]), int(bounds[k + 1]), a.fps, segs[k]) for k in range(a.workers)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        pool.map(worker, jobs)
    lst = OUT / "_segments.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in segs))
    final = OUT / a.name
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True, cwd=OUT)
    for s in segs:
        s.unlink()
    lst.unlink()
    meta = {"frames": n, "fps": a.fps, "duration_s": round(n / a.fps, 2), "render_s": round(time.time() - t0, 1), "size": SIZE}
    (OUT / "render_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {final}  {n} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
