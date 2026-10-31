"""
Keynote-style overlay for the simulation video: a single frosted-glass card on
the right (blur is applied by render.py), SF Pro type, capsule bars, and short
captions bottom-left. Drawn with matplotlib into an RGBA layer; artists are
created once and updated per frame.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

W, H, DPI = 1920, 1080, 100
CARD = (1452, 64, 404, 952)          # x, y, w, h in pixels (top-left origin)
WHITE, LABEL, SECOND, TERTIARY = "#f5f5f7", "#e5e5ea", "#a1a1a6", "#6e6e73"
ORANGE, CYAN, GREEN, RED = "#ff9f0a", "#64d2ff", "#30d158", "#ff453a"


def _family(*names):
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in names if n in have] + ["DejaVu Sans"]


DISPLAY = _family("SF Pro Display", "Helvetica Neue")
TEXT = _family("SF Pro Text", "SF Pro Display", "Helvetica Neue")


def X(px):
    return px / W


def Y(py):
    return 1 - py / H


class Capsule:
    """Rounded progress bar with optional threshold tick."""

    def __init__(self, fig, x, y, w, h=6, track=(1, 1, 1, 0.14), tick=None):
        self.fig, self.x, self.y, self.w, self.h = fig, x, y, w, h
        r = h / 2 / W
        fig.patches.append(FancyBboxPatch((X(x), Y(y + h)), X(w), h / H, boxstyle=f"round,pad=0,rounding_size={r}",
                                          transform=fig.transFigure, facecolor=track, lw=0))
        self.fill = FancyBboxPatch((X(x), Y(y + h)), X(h), h / H, boxstyle=f"round,pad=0,rounding_size={r}",
                                   transform=fig.transFigure, facecolor=WHITE, lw=0)
        fig.patches.append(self.fill)
        if tick is not None:
            fig.lines.append(matplotlib.lines.Line2D([X(x + w * tick)] * 2, [Y(y - 4), Y(y + h + 4)], transform=fig.transFigure,
                                                     color=WHITE, lw=1.2, alpha=0.8))

    def set(self, frac, color):
        self.fill.set_width(X(max(self.h, self.w * float(np.clip(frac, 0, 1)))))
        self.fill.set_facecolor(color)


class HUD:
    def __init__(self, thr=0.8, name="Volition"):
        plt.rcParams.update({"font.family": TEXT, "text.color": WHITE, "axes.edgecolor": "none"})
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        self.fig.patch.set_alpha(0.0)
        f = self.fig
        cx, cy, cw, ch = CARD
        L, R = cx + 28, cx + cw - 28
        shadow = [pe.withSimplePatchShadow(offset=(0, -1), shadow_rgbFace="black", alpha=0.35)]

        # top-left: wordmark + state
        f.text(X(64), Y(78), name, fontsize=22, family=DISPLAY, weight="semibold", color=WHITE, va="center")
        self.t_state = f.text(X(62), Y(140), "", fontsize=46, family=DISPLAY, weight="medium", color=WHITE, va="center", path_effects=shadow)
        self.t_sub = f.text(X(64), Y(190), "", fontsize=17, family=TEXT, color=SECOND, va="center")

        def header(y, title, right):
            f.text(X(L), Y(y), title, fontsize=16, family=DISPLAY, weight="semibold", color=WHITE, va="center")
            f.text(X(R), Y(y), right, fontsize=12.5, family=TEXT, color=SECOND, va="center", ha="right")

        def hair(y):
            f.lines.append(matplotlib.lines.Line2D([X(L), X(R)], [Y(y)] * 2, transform=f.transFigure, color=(1, 1, 1, 0.12), lw=1))

        # card: brain signal
        header(cy + 34, "Brain signal", "EEG · 500 Hz")
        self.ax_eeg = f.add_axes([X(L + 30), Y(cy + 182), X(R - L - 30), 128 / H])
        self.ax_eeg.set_axis_off()
        self.ax_eeg.set_ylim(-2.6, 0.6)
        for k, name in enumerate(("C3", "Cz", "C4")):
            f.text(X(L), Y(cy + 92 + k * 38), name, fontsize=12, color=TERTIARY, va="center")
        self.eeg_lines = [self.ax_eeg.plot([], [], lw=1.0, color=c, alpha=a, solid_capstyle="round")[0]
                          for c, a in ((WHITE, 0.55), (CYAN, 0.95), (WHITE, 0.55))]
        self.eeg_clip = self.ax_eeg.axvspan(0, 0, color=RED, alpha=0.0, lw=0)
        self.rows = {}
        for k, (cls, label) in enumerate((("REST", "Rest"), ("FEET_MI", "Feet  ·  stand, walk"), ("HANDS_MI", "Hands  ·  stop, sit"))):
            y = cy + 220 + k * 50
            t = f.text(X(L), Y(y), label, fontsize=15, color=LABEL, va="center")
            v = f.text(X(R), Y(y), "", fontsize=15, family=DISPLAY, color=SECOND, va="center", ha="right")
            bar = Capsule(f, L, y + 14, R - L, tick=thr)
            self.rows[cls] = (t, v, bar)
        self.t_bci = f.text(X(L), Y(cy + 380), "", fontsize=13, color=SECOND, va="center")
        hair(cy + 408)

        # card: joints
        header(cy + 440, "Joints", "torque and live limit")
        self.joints = {}
        for k, (j, label) in enumerate((("hip_L", "Left hip"), ("hip_R", "Right hip"), ("knee_L", "Left knee"), ("knee_R", "Right knee"))):
            y = cy + 478 + k * 86
            f.text(X(L), Y(y), label, fontsize=15, color=LABEL, va="center")
            v = f.text(X(R), Y(y), "", fontsize=18, family=DISPLAY, weight="medium", color=WHITE, va="center", ha="right")
            tag = f.text(X(R - 96), Y(y), "", fontsize=12, color=RED, va="center", ha="right", weight="semibold")
            ax = f.add_axes([X(L), Y(y + 62), X(R - L), 44 / H])
            ax.set_axis_off()
            ax.set_ylim(-125, 125)
            band = ax.fill_between([0, 1], [0, 0], [0, 0], color=(1, 1, 1, 0.08), lw=0)
            ax.axhline(0, color=(1, 1, 1, 0.12), lw=0.8)
            line, = ax.plot([], [], color=WHITE, lw=1.4, solid_capstyle="round")
            dot = ax.scatter([], [], s=16, color=WHITE, zorder=5)
            satd = ax.scatter([], [], s=6, color=RED, zorder=6)
            self.joints[j] = dict(v=v, tag=tag, ax=ax, band=band, line=line, dot=dot, sat=satd)
        hair(cy + 822)

        # card: battery
        header(cy + 852, "Battery", "13S2P · 468 Wh")
        self.t_volt = f.text(X(L), Y(cy + 896), "", fontsize=28, family=DISPLAY, weight="medium", color=WHITE, va="center")
        self.t_soc = f.text(X(R), Y(cy + 896), "", fontsize=28, family=DISPLAY, weight="medium", color=WHITE, va="center", ha="right")
        self.soc_bar = Capsule(f, L, cy + 918, R - L, h=6)
        self.t_pwr = f.text(X(L), Y(cy + 940), "", fontsize=12.5, color=SECOND, va="center")

        # captions, title cards, footer
        self.t_cap = f.text(X(64), Y(986), "", fontsize=34, family=DISPLAY, weight="medium", color=WHITE, va="center", path_effects=shadow)
        self.t_title = f.text(X(960), Y(505), "", fontsize=104, family=DISPLAY, weight="semibold", color=WHITE, ha="center", va="center")
        self.t_title2 = f.text(X(960), Y(595), "", fontsize=30, family=DISPLAY, color=SECOND, ha="center", va="center")
        f.text(X(64), Y(1038), "Simulation with synthetic EEG. Concept design, not a medical device.", fontsize=11.5, color=TERTIARY, va="center")

    def draw(self, d):
        self.t_state.set_text(d["state"])
        self.t_sub.set_text(d["state_sub"])

        t_e, eeg = d["eeg_t"], d["eeg"]
        for k, line in enumerate(self.eeg_lines):
            line.set_data(t_e, eeg[:, k] - k)
        self.ax_eeg.set_xlim(d["t"] - 4.0, d["t"])
        self.eeg_clip.remove()
        a, b = d["clip_span"] if d["clip_span"] else (0, 0)
        self.eeg_clip = self.ax_eeg.axvspan(a, b, color=RED, alpha=0.18 if d["clip_span"] else 0.0, lw=0)
        colors = {"REST": SECOND, "FEET_MI": ORANGE, "HANDS_MI": CYAN}
        for cls, (t, v, bar) in self.rows.items():
            ev, p = d["evidence"][cls], d["prob"][cls]
            bar.set(ev, colors[cls])
            v.set_text(f"{p * 100:.0f}%")
            v.set_color(WHITE if (cls != "REST" and ev > 0.5) else SECOND)
        self.t_bci.set_text(d["bci_line"][0])
        self.t_bci.set_color(d["bci_line"][1])

        for j, art in self.joints.items():
            ax, tt = art["ax"], d["tq_t"]
            art["band"].remove()
            art["band"] = ax.fill_between(tt, d["lim_lo"][j], d["lim_hi"][j], color=(1, 1, 1, 0.08), lw=0)
            art["line"].set_data(tt, d["out"][j])
            if len(tt):
                art["dot"].set_offsets([[tt[-1], d["out"][j][-1]]])
            m = d["satm"][j]
            art["sat"].set_offsets(np.c_[tt[m], d["out"][j][m]] if m.any() else np.empty((0, 2)))
            ax.set_xlim(d["t"] - 5.0, d["t"])
            reason = d["reason_now"][j]
            art["v"].set_text(f"{abs(d['out_now'][j]):.0f} N·m")
            art["v"].set_color(RED if reason else WHITE)
            art["tag"].set_text(d["reason_txt"][j] if reason else "")

        self.t_volt.set_text(f"{d['vbus']:.1f} V")
        self.t_soc.set_text(f"{d['soc'] * 100:.0f}%")
        self.soc_bar.set(d["soc"], GREEN)
        self.t_pwr.set_text(d["pwr_line"])

        cap, ca = d.get("caption", ("", 0.0))
        self.t_cap.set_text(cap)
        self.t_cap.set_alpha(ca)
        self.t_title.set_text(d.get("title", ""))
        self.t_title.set_fontsize(d.get("title_size", 104))
        self.t_title2.set_text(d.get("title2", ""))
        self.t_title.set_alpha(d.get("title_alpha", 0.0))
        self.t_title2.set_alpha(d.get("title_alpha", 0.0))

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())
