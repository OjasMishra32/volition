"""
Motor-imagery BCI: a synthetic 16-channel EEG source model and the online decoder.

Decoder chain (application processor, EEG_CFG.update_hz):
  60 Hz notch -> 8-30 Hz band-pass -> EOG regression -> common average reference
  -> artifact gate (ADC clipping, flat-line, amplitude) on the 1 s window
  -> log band power, mu 8-13 Hz and beta 16-26 Hz, on 11 sensorimotor channels
  -> shrinkage LDA over REST / FEET_MI / HANDS_MI
  -> leaky evidence integrator -> threshold + dwell + refractory -> intent events.

The synthetic EEG is only a test signal for the pipeline: mu/beta generators over
the hand and foot areas whose power drops (ERD) during imagery, 1/f background,
common-mode drift, line noise, blinks and saccades, gait-locked motion artefacts
and an optional electrode pop that drives the ADS1299 into clipping.
"""
import numpy as np
from scipy import signal

from params import EEG_CFG

FS = EEG_CFG.fs
HOP = FS / EEG_CFG.update_hz
CH = list(EEG_CFG.channels)
N_CH = len(CH)
EEG_I = [i for i, c in enumerate(CH) if not c.endswith("EOG")]
EOG_I = [CH.index("HEOG"), CH.index("VEOG")]
FEAT_CH = ["FC3", "FCz", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CPz", "CP4"]
FEAT_I = [EEG_I.index(CH.index(c)) for c in FEAT_CH]
DISPLAY_CH = ["C3", "Cz", "C4"]
DISPLAY_I = [EEG_I.index(CH.index(c)) for c in DISPLAY_CH]
CLASSES = list(EEG_CFG.classes)
FULL_SCALE = EEG_CFG.adc_full_scale          # volts
LSB = 2 * FULL_SCALE / 2 ** 24
AMP_REJECT_UV = 120.0
FLAT_UV = 0.05

# electrode positions (x = right, y = anterior), arbitrary units
POS = {"FC3": (-0.5, 0.5), "FCz": (0, 0.5), "FC4": (0.5, 0.5), "C5": (-0.75, 0), "C3": (-0.5, 0),
       "C1": (-0.25, 0), "Cz": (0, 0), "C2": (0.25, 0), "C4": (0.5, 0), "C6": (0.75, 0),
       "CP3": (-0.5, -0.5), "CPz": (0, -0.5), "CP4": (0.5, -0.5), "Pz": (0, -1.0)}
SOURCES = {"hand_L": (-0.5, 0.0), "hand_R": (0.5, 0.0), "feet": (0.0, 0.0)}
ERD = {"REST": {"hand_L": 1.0, "hand_R": 1.0, "feet": 1.0},
       "FEET_MI": {"hand_L": 1.06, "hand_R": 1.06, "feet": 0.47},
       "HANDS_MI": {"hand_L": 0.47, "hand_R": 0.47, "feet": 1.02}}
ERD_JITTER = 0.25              # trial-to-trial variability of imagery strength
STATE_SIGMA, STATE_TAU = 0.12, 4.0   # slow drift of rhythm power (alertness, fatigue)
MU_UV, BETA_UV, ALPHA_UV, PINK_UV, COMMON_UV, LINE_UV = 9.0, 4.0, 7.0, 8.0, 10.0, 4.0
BLINK_LEAK = {"FC": 0.22, "C": 0.08, "CP": 0.03, "Pz": 0.01}


def _spread(xy, sigma):
    out = np.zeros(N_CH)
    for i, c in enumerate(CH):
        if c in POS:
            d2 = (POS[c][0] - xy[0]) ** 2 + (POS[c][1] - xy[1]) ** 2
            out[i] = np.exp(-d2 / (2 * sigma ** 2))
    return out


def _leak(table):
    out = np.zeros(N_CH)
    for i, c in enumerate(CH):
        for prefix in sorted(table, key=len, reverse=True):
            if c.startswith(prefix) and c in POS:
                out[i] = table[prefix]
                break
    return out


class _Osc:
    """Narrow-band oscillator: AR(2) resonator driven by white noise, unit variance."""

    def __init__(self, f, bw, rng):
        r = np.exp(-np.pi * bw / FS)
        self.a = [1.0, -2 * r * np.cos(2 * np.pi * f / FS), r * r]
        self.zi = np.zeros(2)
        x = signal.lfilter([1.0], self.a, rng.standard_normal(FS * 40))
        self.scale = 1.0 / np.std(x[FS * 5:])

    def run(self, w):
        y, self.zi = signal.lfilter([1.0], self.a, w, zi=self.zi)
        return y * self.scale


class EEGGenerator:
    PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]

    def __init__(self, seed=0):
        rng = self.rng = np.random.default_rng(seed)
        self.n = 0
        self.osc = {(s, b): _Osc(f, bw, rng) for s in SOURCES
                    for b, f, bw in (("mu", 10.5, 2.0), ("beta", 21.0, 3.0))}
        self.alpha = _Osc(10.0, 1.5, rng)
        self.common = _Osc(4.0, 6.0, rng)
        self.spread = {s: _spread(xy, 0.27) for s, xy in SOURCES.items()}
        self.alpha_spread = _spread((0.0, -1.0), 0.45)
        self.eeg_mask = np.isin(np.arange(N_CH), EEG_I).astype(float)
        self.pink_zi = np.zeros((3, N_CH))
        x = signal.lfilter(self.PINK_B, self.PINK_A, rng.standard_normal(FS * 40))
        self.pink_scale = 1.0 / np.std(x[FS * 5:])
        self.gain = {s: 1.0 for s in SOURCES}
        self.mental = "REST"
        self.jitter = {s: 1.0 for s in SOURCES}
        self.state = 0.0
        self.offset = rng.uniform(-25e-3, 25e-3, N_CH)
        self.line_amp = rng.uniform(0.5, 1.5, N_CH) * LINE_UV
        self.line_ph = rng.uniform(0, 2 * np.pi, N_CH)
        self.blink_leak = _leak(BLINK_LEAK)
        self.blink_leak[CH.index("VEOG")] = 1.0
        self.blink_leak[CH.index("HEOG")] = 0.05
        self.sacc_leak = np.zeros(N_CH)
        for c, v in (("HEOG", 1.0), ("FC3", 0.08), ("FC4", -0.08), ("C5", 0.05), ("C6", -0.05)):
            self.sacc_leak[CH.index(c)] = v
        self.motion_w = rng.normal(1.0, 0.4, N_CH)
        self.events = []                 # (t0, waveform (n, N_CH))
        self.next_blink = rng.exponential(5.0)
        self.next_sacc = rng.exponential(8.0)
        self.pops = []                   # (t0, channel index, amplitude V)

    def schedule_pop(self, t0, ch="C4", amp=0.26):
        self.pops.append((t0, CH.index(ch), amp))

    def add_heel_strike(self, t0):
        """Cable sway + muscle burst after a heel strike, scaled per channel."""
        rng = self.rng
        tt = np.arange(int(0.6 * FS)) / FS
        w = rng.normal(1.0, 0.35, N_CH) * self.eeg_mask
        wave = 35.0 * np.exp(-tt / 0.10) * np.sin(2 * np.pi * 6.5 * tt)
        b, a = signal.butter(2, (18, 40), "bandpass", fs=FS)
        emg = signal.lfilter(b, a, rng.standard_normal((len(tt), N_CH)), axis=0) * 6.0
        emg *= (tt < 0.15)[:, None]
        self.events.append((t0, wave[:, None] * w[None] + emg * self.eeg_mask))

    def generate(self, n, mental="REST", moving=False, step_hz=0.0):
        """Next n samples as raw ADC volts (n, 16) and the per-sample clip flag."""
        rng = self.rng
        t = (self.n + np.arange(n)) / FS
        t_end = t[-1] + 1.0 / FS
        if mental != self.mental:
            self.mental = mental
            self.jitter = {s: float(np.clip(rng.normal(1.0, ERD_JITTER), 0.2, 1.8)) for s in SOURCES}
        dt = n / FS
        self.state = self.state * (1 - dt / STATE_TAU) + STATE_SIGMA * np.sqrt(2 * dt / STATE_TAU) * rng.standard_normal()
        ramp = 1.0 - np.exp(-np.arange(1, n + 1) / FS / 0.45)
        x = np.zeros((n, N_CH))
        for s in SOURCES:
            target = 1.0 + (ERD[mental][s] - 1.0) * self.jitter[s]
            g = self.gain[s] + (target - self.gain[s]) * ramp
            self.gain[s] = g[-1]
            for band, amp in (("mu", MU_UV), ("beta", BETA_UV)):
                src = self.osc[(s, band)].run(rng.standard_normal(n)) * amp * g * np.exp(self.state)
                x += src[:, None] * self.spread[s][None]
        x += (self.alpha.run(rng.standard_normal(n)) * ALPHA_UV)[:, None] * self.alpha_spread[None]
        x += (self.common.run(rng.standard_normal(n)) * COMMON_UV)[:, None] * self.eeg_mask[None]
        pink, self.pink_zi = signal.lfilter(self.PINK_B, self.PINK_A, rng.standard_normal((n, N_CH)),
                                            axis=0, zi=self.pink_zi)
        x += pink * self.pink_scale * PINK_UV
        x += self.line_amp * np.sin(2 * np.pi * 60.0 * t[:, None] + self.line_ph)
        while self.next_blink < t_end:
            tt = np.arange(int(0.8 * FS)) / FS
            wave = rng.uniform(120, 200) * np.exp(-0.5 * ((tt - 0.3) / 0.08) ** 2)
            self.events.append((self.next_blink, wave[:, None] * self.blink_leak[None]))
            self.next_blink += rng.exponential(5.0)
        while self.next_sacc < t_end:
            d = rng.uniform(0.4, 1.5)
            tt = np.arange(int((d + 0.4) * FS)) / FS
            sh = (1 / (1 + np.exp(-(tt - 0.1) / 0.01)) - 1 / (1 + np.exp(-(tt - 0.1 - d) / 0.01)))
            wave = rng.choice([-1, 1]) * rng.uniform(30, 60) * sh
            self.events.append((self.next_sacc, wave[:, None] * self.sacc_leak[None]))
            self.next_sacc += rng.exponential(8.0)
        if moving:
            sway = 18.0 * np.sin(2 * np.pi * max(step_hz, 0.3) * t) + 8.0 * np.sin(4 * np.pi * max(step_hz, 0.3) * t + 1.0)
            x += sway[:, None] * self.motion_w[None] * self.eeg_mask[None]
        keep = []
        for t0, wave in self.events:
            i0 = int(round((t0 - t[0]) * FS))
            a, b = max(i0, 0), min(i0 + len(wave), n)
            if b > a:
                x[a:b] += wave[a - i0:b - i0]
            if i0 + len(wave) > n:
                keep.append((t0, wave))
        self.events = keep
        v = x * 1e-6 + self.offset
        for t0, ci, amp in self.pops:
            m = t >= t0
            v[m, ci] += amp * np.exp(-(t[m] - t0) / 2.5)
        clip = np.abs(v) >= EEG_CFG.clip_frac * FULL_SCALE
        v = np.round(np.clip(v, -FULL_SCALE, FULL_SCALE - LSB) / LSB) * LSB
        self.n += n
        return v, clip.any(1)


class ShrinkageLDA:
    """Multi-class LDA with Ledoit-Wolf shrinkage of the pooled covariance."""

    def fit(self, X, y, n_classes=3):
        d = X.shape[1]
        self.mu = np.array([X[y == k].mean(0) for k in range(n_classes)])
        Xc = X - self.mu[y]
        S = Xc.T @ Xc / len(X)
        nu = np.trace(S) / d
        num = np.sum((np.einsum("ni,nj->nij", Xc, Xc) - S) ** 2) / len(X) ** 2
        den = np.sum((S - nu * np.eye(d)) ** 2)
        self.shrink = float(np.clip(num / den, 0, 1))
        cov = (1 - self.shrink) * S + self.shrink * nu * np.eye(d)
        self.W = np.linalg.solve(cov, self.mu.T)
        prior = np.bincount(y, minlength=n_classes) / len(y)
        self.b = -0.5 * np.sum(self.mu.T * self.W, 0) + np.log(prior)
        return self

    def proba(self, X):
        z = np.atleast_2d(X) @ self.W + self.b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)


class Decoder:
    def __init__(self):
        self.sos_notch = signal.tf2sos(*signal.iirnotch(EEG_CFG.notch_hz, 30.0, FS))
        self.sos_bp = signal.butter(4, EEG_CFG.band, "bandpass", fs=FS, output="sos")
        self.sos_mu = signal.butter(4, (8.0, 13.0), "bandpass", fs=FS, output="sos")
        self.sos_beta = signal.butter(4, (16.0, 26.0), "bandpass", fs=FS, output="sos")
        self.B = np.zeros((2, len(EEG_I)))
        self.lda = None
        self.f_mean = self.f_sd = None
        self.reset()

    def reset(self):
        self.z_notch = self.z_bp = None
        self.z_mu = np.zeros((self.sos_mu.shape[0], 2, len(FEAT_I)))
        self.z_beta = np.zeros((self.sos_beta.shape[0], 2, len(FEAT_I)))
        self.buf_pw = np.zeros((0, 2 * len(FEAT_I)))
        self.buf_raw = np.zeros((0, N_CH))
        self.buf_clip = np.zeros(0, bool)
        self.buf_car = np.zeros((0, len(EEG_I)))
        self.E = np.array([1.0, 0.0, 0.0])
        self.above = [None, None, None]
        self.refractory_until = -1e9
        self.recent = []

    # -- signal chain ----------------------------------------------------------
    def _pre(self, raw_v):
        uv = raw_v * 1e6
        if self.z_notch is None:
            self.z_notch = signal.sosfilt_zi(self.sos_notch)[:, :, None] * uv[0][None, None, :]
            self.z_bp = signal.sosfilt_zi(self.sos_bp)[:, :, None] * uv[0][None, None, :]
        y, self.z_notch = signal.sosfilt(self.sos_notch, uv, axis=0, zi=self.z_notch)
        y, self.z_bp = signal.sosfilt(self.sos_bp, y, axis=0, zi=self.z_bp)
        return y

    def _post(self, y):
        eeg = y[:, EEG_I] - y[:, EOG_I] @ self.B
        car = eeg - eeg.mean(1, keepdims=True)
        f = car[:, FEAT_I]
        mu, self.z_mu = signal.sosfilt(self.sos_mu, f, axis=0, zi=self.z_mu)
        be, self.z_beta = signal.sosfilt(self.sos_beta, f, axis=0, zi=self.z_beta)
        return car, np.hstack([mu ** 2, be ** 2])

    def push(self, raw_v, clip):
        """Filter a chunk, update the 1 s buffers; returns C3/Cz/C4 after CAR (uV)."""
        car, pw = self._post(self._pre(raw_v))
        self.buf_pw = np.vstack([self.buf_pw, pw])[-FS:]
        self.buf_raw = np.vstack([self.buf_raw, raw_v])[-FS:]
        self.buf_clip = np.concatenate([self.buf_clip, clip])[-FS:]
        self.buf_car = np.vstack([self.buf_car, car])[-FS:]
        return car[:, DISPLAY_I]

    def window(self):
        if len(self.buf_pw) < FS:
            return None, "warm-up"
        if self.buf_clip.any():
            return None, "clip"
        if (self.buf_raw[:, EEG_I].std(0) * 1e6 < FLAT_UV).any():
            return None, "flat-line"
        if np.abs(self.buf_car).max() > AMP_REJECT_UV:
            return None, "amplitude"
        return np.log(self.buf_pw.mean(0) + 1e-3), ""

    def classify(self, f):
        return self.lda.proba((f - self.f_mean) / self.f_sd)[0]

    def update(self, t):
        """One decoder tick: probabilities, evidence, validity and an optional event."""
        f, reason = self.window()
        valid = f is not None
        probs = self.classify(f) if valid else np.array([1.0, 0.0, 0.0])
        a = 1.0 - np.exp(-1.0 / (EEG_CFG.update_hz * EEG_CFG.evidence_tau_s))
        if valid:
            self.E += a * (probs - self.E)
        else:                                   # rejected window: decay toward REST
            self.E += a * (np.array([1.0, 0.0, 0.0]) - self.E)
        self.recent = (self.recent + [probs])[-int(EEG_CFG.update_hz):]
        event = None
        for k in (1, 2):
            if self.E[k] >= EEG_CFG.evidence_threshold and t >= self.refractory_until:
                if self.above[k] is None:
                    self.above[k] = t
                if t - self.above[k] >= EEG_CFG.min_dwell_s - 1e-9:
                    p = float(np.mean([r[k] for r in self.recent]))
                    event = (CLASSES[k], p)
                    self.E[:] = (1.0, 0.0, 0.0)
                    self.above = [None, None, None]
                    self.refractory_until = t + EEG_CFG.refractory_s
                    break
            else:
                self.above[k] = None
        return dict(probs=probs, evidence=self.E.copy(), valid=valid, reason=reason, event=event)

    # -- offline helpers (calibration) -----------------------------------------
    def offline_features(self, raw, clip, ends):
        """Features for 1 s windows ending at sample indices `ends` (B already set)."""
        self.reset()
        y = self._pre(raw)
        car, pw = self._post(y)
        cs = np.vstack([np.zeros((1, pw.shape[1])), np.cumsum(pw, 0)])
        F = np.log((cs[ends] - cs[ends - FS]) / FS + 1e-3)
        cc = np.concatenate([[0], np.cumsum(clip)])
        peak = np.abs(car).max(1)
        amp = np.array([peak[e - FS:e].max() for e in ends])
        ok = ((cc[ends] - cc[ends - FS]) == 0) & (amp <= AMP_REJECT_UV)
        return F, ok

    def fit_eog(self, raw):
        self.reset()
        y = self._pre(raw)[FS:]
        self.B = np.linalg.lstsq(y[:, EOG_I], y[:, EEG_I], rcond=None)[0]

    def fit(self, F, y):
        self.f_mean, self.f_sd = F.mean(0), F.std(0) + 1e-9
        self.lda = ShrinkageLDA().fit((F - self.f_mean) / self.f_sd, y)
        return self


def record(gen, plan, chunk=int(HOP)):
    """plan: [(seconds, mental, moving, label or -1)] -> raw, clip, per-sample labels, trial ids."""
    raws, clips, labels, trials = [], [], [], []
    for trial, (sec, mental, moving, label) in enumerate(plan):
        n = int(round(sec * FS))
        done = 0
        while done < n:
            m = min(chunk, n - done)
            v, c = gen.generate(m, mental, moving, step_hz=1.0 / 1.5)
            if moving and gen.rng.random() < m / (0.75 * FS):
                gen.add_heel_strike((gen.n) / FS)
            raws.append(v)
            clips.append(c)
            done += m
        labels.append(np.full(n, label))
        trials.append(np.full(n, trial))
    return np.vstack(raws), np.concatenate(clips), np.concatenate(labels), np.concatenate(trials)


def calibrate(seed=11, trials_per_class=30, task_s=4.0, rest_s=2.0, train_frac=0.7):
    """Synthetic cue-based calibration session; returns (decoder, report)."""
    rng = np.random.default_rng(seed)
    gen = EEGGenerator(seed)
    order = rng.permutation(np.repeat(np.arange(3), trials_per_class))
    plan = []
    for k in order:
        plan.append((rest_s + rng.uniform(0, 1.0), "REST", False, -1))
        plan.append((task_s, CLASSES[k], False, int(k)))
    raw, clip, lab, tri = record(gen, plan)
    dec = Decoder()
    dec.fit_eog(raw)
    hop = int(HOP)
    ends = np.arange(FS, len(raw), hop)
    # a window is labelled when it lies entirely inside a task period, >= 0.5 s after the cue
    cue_start = {}
    for i in np.flatnonzero(np.diff(np.concatenate([[-2], tri])) != 0):
        cue_start[tri[i]] = i
    keep, y, trial_of = [], [], []
    for e in ends:
        s = e - FS
        if lab[s] >= 0 and lab[e - 1] == lab[s] and tri[s] == tri[e - 1] and s - cue_start[tri[s]] >= FS // 2:
            keep.append(e)
            y.append(lab[s])
            trial_of.append(tri[s])
    keep, y, trial_of = np.array(keep), np.array(y), np.array(trial_of)
    F, ok = dec.offline_features(raw, clip, keep)
    F, y, trial_of = F[ok], y[ok], trial_of[ok]
    task_trials = np.unique(trial_of)
    split = task_trials[int(train_frac * len(task_trials))]
    tr, te = trial_of < split, trial_of >= split
    dec.fit(F[tr], y[tr])
    pred = dec.lda.proba((F[te] - dec.f_mean) / dec.f_sd).argmax(1)
    conf = np.zeros((3, 3), int)
    for a, b in zip(y[te], pred):
        conf[a, b] += 1
    acc = float((pred == y[te]).mean())
    dec.fit(F, y)                 # final model uses every calibration trial
    report = dict(windows_train=int(tr.sum()), windows_test=int(te.sum()), heldout_window_accuracy=acc,
                  confusion_rows_true_cols_pred=conf.tolist(), classes=CLASSES,
                  lda_shrinkage=dec.lda.shrink, chance=1 / 3)
    return dec, report


def false_activation_test(dec, seed=21, minutes_seated=3.0, minutes_walking=2.0):
    """Run the online decoder on pure REST and count intent events per minute."""
    out = {}
    for name, minutes, moving in (("seated", minutes_seated, False), ("walking", minutes_walking, True)):
        gen = EEGGenerator(seed + (7 if moving else 0))
        dec.reset()
        n_ticks = int(minutes * 60 * EEG_CFG.update_hz)
        events, t, next_hs = 0, 0.0, 0.75
        acc = 0.0
        for _ in range(n_ticks):
            acc += HOP
            m = int(acc)
            acc -= m
            if moving:
                while next_hs < t + m / FS:
                    gen.add_heel_strike(next_hs)
                    next_hs += 0.75
            v, c = gen.generate(m, "REST", moving, step_hz=1 / 1.5)
            dec.push(v, c)
            t += m / FS
            if dec.update(t)["event"] is not None:
                events += 1
        out[name] = dict(minutes=minutes, events=events, per_minute=events / minutes)
    dec.reset()
    return out
