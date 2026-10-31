"""
EXA-100 actuator, servo drive and battery/DC-bus model, including the full
output-torque saturation chain used by the joint controllers.

Torques here are gearbox output torques ("tau_gear" = N * eta * motor torque),
in the joint-flexion sign convention.  Efficiency depends on power direction:
motoring  tau_gear = N * eta_fwd * tau_m,  back-driven  tau_gear = N * tau_m / eta_bwd.

Saturation reason codes (per joint, per tick):
  0 none   1 software / gear peak   2 drive current   3 voltage (dq circle or
  low-battery derate)   4 thermal current derate   5 torque slew limit
"""
import numpy as np

from params import ACT, BATT

N = ACT.ratio
LAMBDA = (2.0 / 3.0) * ACT.kt / ACT.pole_pairs     # PM flux linkage (amplitude-invariant dq)
REASONS = ("none", "software/gear peak", "drive current", "voltage", "thermal", "slew")
P_AUX = 32.0                    # W: safety MCU, app processor (BCI), EEG amp, IMUs, fans
DRIVE_IDLE_W, DRIVE_LOSS_FRAC = 2.5, 0.03
PEAK = min(ACT.tau_peak_sw, ACT.gear_repeated_peak)


def voltage_derate(v_bus):
    """Peak-torque scale: 1 above the UVLO warning, 0.6 at the forced-sit threshold."""
    x = (v_bus - BATT.uvlo_sit_v) / (BATT.uvlo_warn_v - BATT.uvlo_sit_v)
    return float(np.clip(0.6 + 0.4 * x, 0.6, 1.0))


def current_limit(t_wind):
    x = np.clip((t_wind - ACT.t_derate_start) / (ACT.t_max - ACT.t_derate_start), 0.0, 1.0)
    return ACT.i_peak - (ACT.i_peak - ACT.i_cont) * x


def gear_torque(iq, w_out):
    tm = ACT.kt * iq
    return np.where(tm * w_out >= 0, N * ACT.eff_forward * tm, N * tm / ACT.eff_backward)


def iq_for(tau, w_out):
    tm = np.where(tau * w_out >= 0, tau / (N * ACT.eff_forward), tau * ACT.eff_backward / N)
    return tm / ACT.kt


def voltage_window(w_out, v_bus):
    """Feasible i_q interval inside the SVPWM voltage circle with i_d = 0."""
    we = ACT.pole_pairs * N * np.asarray(w_out, float)
    vmax = v_bus * ACT.svpwm_utilization
    a = ACT.r_phase ** 2 + (we * ACT.l_phase) ** 2
    b = 2 * ACT.r_phase * we * LAMBDA
    c = (we * LAMBDA) ** 2 - vmax ** 2
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    mid = -b / (2 * a)
    lo = np.where(disc >= 0, (-b - sq) / (2 * a), mid)
    hi = np.where(disc >= 0, (-b + sq) / (2 * a), mid)
    return lo, hi


def torque_limits(w_out, v_bus, t_wind):
    """Static output-torque envelope [lo, hi] and the binding reason on each side."""
    w = np.atleast_1d(np.asarray(w_out, float))
    t_wind = np.broadcast_to(t_wind, w.shape)
    dv = voltage_derate(v_bus)
    hi = np.full(w.shape, PEAK * dv)
    lo = -hi
    r = 3 if dv < 1.0 else 1
    rhi = np.full(w.shape, r)
    rlo = np.full(w.shape, r)
    ilim = current_limit(t_wind)
    ri = np.where(t_wind > ACT.t_derate_start, 4, 2)
    vlo, vhi = voltage_window(w, v_bus)
    for cand_hi, cand_lo, reason in ((gear_torque(ilim, w), gear_torque(-ilim, w), ri),
                                     (gear_torque(vhi, w), gear_torque(vlo, w), 3)):
        m = cand_hi < hi
        hi = np.where(m, cand_hi, hi)
        rhi = np.where(m, reason, rhi)
        m = cand_lo > lo
        lo = np.where(m, cand_lo, lo)
        rlo = np.where(m, reason, rlo)
    return lo, hi, rlo, rhi


def saturate(tau_cmd, tau_prev, w_out, v_bus, t_wind, dt):
    """Full chain: slew limit, then the static envelope. Returns (tau, reason, lo, hi)."""
    lo, hi, rlo, rhi = torque_limits(w_out, v_bus, t_wind)
    step = ACT.torque_slew_limit * dt
    tau = np.clip(tau_cmd, tau_prev - step, tau_prev + step)
    reason = np.where(tau != tau_cmd, 5, 0)
    out = np.clip(tau, lo, hi)
    reason = np.where(out > tau, rlo, np.where(out < tau, rhi, reason))
    return out, reason, lo, hi


def electrical(tau_gear, w_out):
    """Phase current, electrical input power (incl. copper and drive losses), copper loss."""
    iq = iq_for(tau_gear, w_out)
    p_cu = 1.5 * ACT.r_phase * iq ** 2
    p = ACT.kt * iq * N * w_out + p_cu
    p = p + DRIVE_IDLE_W + DRIVE_LOSS_FRAC * np.abs(p)
    return iq, p, p_cu


def thermal_step(t_wind, p_cu, dt):
    return t_wind + dt * (p_cu - (t_wind - ACT.t_ambient) / ACT.r_th) / ACT.c_th


class Battery:
    """13S2P pack + DC bus: OCV(SoC), internal resistance, BMS charge-current limit,
    bus capacitance and a brake chopper with hysteresis."""

    def __init__(self, soc=0.62):
        self.soc = soc
        self.v_bus = self.ocv()
        self.v_cap = self.v_bus
        self.chopper = False
        self.i_batt = 0.0

    def ocv(self):
        return BATT.series * float(np.interp(self.soc, BATT.ocv_soc, BATT.ocv_v))

    def charge_limit(self):
        return BATT.i_charge_max * float(np.clip((1.0 - self.soc) / 0.05, 0.0, 1.0))

    def step(self, p_load, dt):
        i_load = p_load / self.v_bus
        if self.chopper:
            i_load += self.v_bus / BATT.chopper_r
        i_b = max(i_load, -self.charge_limit())
        v_batt = self.ocv() - BATT.r_pack * i_b
        surplus = i_b - i_load                     # regen the battery refused -> bus caps
        if surplus > 1e-9:
            self.v_cap += surplus * dt / BATT.bus_capacitance
        else:
            self.v_cap = v_batt
        self.v_bus = max(v_batt, self.v_cap)
        if self.v_bus > BATT.chopper_on_v:
            self.chopper = True
        elif self.v_bus < BATT.chopper_off_v:
            self.chopper = False
        self.soc -= i_b * dt / 3600.0 / BATT.capacity_ah
        self.i_batt = i_b
        return self.v_bus

    @staticmethod
    def soc_for_ocv(v):
        return float(np.interp(v / BATT.series, BATT.ocv_v, BATT.ocv_soc))


def envelope(v_bus, w_out, t_wind=ACT.t_ambient):
    """Torque-speed envelope for plotting (no slew)."""
    lo, hi, _, _ = torque_limits(w_out, v_bus, t_wind)
    return lo, hi
