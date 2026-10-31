"""
Per-joint 1 kHz controller for the four powered joints (JOINTS order).

  tau_cmd = tau_ID(model) + N^2 J_m qdd_ref + friction comp      (feed-forward)
          + Kp e + Kd(J) e_dot + I                                 (PID on output angle)
          + virtual springs beyond the joint ROM                   (soft limits)
  tau_out = saturate(tau_cmd)                                      (full actuator chain)
  dI/dt   = Ki e  (frozen while it would push further into an active limit)
          + (tau_out - tau_cmd) / Tt, applied only to bleed I toward zero  (back-calculation)

Kd is scheduled with the effective inertia the joint sees (stance vs swing) to
keep the error dynamics damped.  The back-calculation only bleeds the integrator
toward zero: most of a saturated command here is model feed-forward, and letting
the integrator cancel that would leave it wound the other way when the limit
releases.  `antiwindup=False` gives the naive PID used in the comparison study.
"""
import numpy as np

from params import ACT, CTRL

KINDS = ("hip", "knee", "hip", "knee")
N2JM = ACT.ratio ** 2 * ACT.j_rotor
ROM = np.array([CTRL.hip_rom, CTRL.knee_rom, CTRL.hip_rom, CTRL.knee_rom])
HARD_STOP = ROM + np.array([-1, 1]) * CTRL.soft_limit_margin   # mechanical stops


def friction(qd):
    return ACT.coulomb_friction_out * np.tanh(qd / 0.05) + ACT.viscous_out * qd


class JointController:
    def __init__(self, dt, antiwindup=True, zeta=0.75):
        self.dt = dt
        self.antiwindup = antiwindup
        self.zeta = zeta
        self.kp = np.array([CTRL.kp[k] for k in KINDS])
        self.kd = np.array([CTRL.kd[k] for k in KINDS])
        self.ki = np.array([CTRL.ki[k] for k in KINDS])
        self.reset()

    def reset(self, tau0=None):
        self.integ = np.zeros(4)
        self.tau_prev = np.zeros(4) if tau0 is None else np.array(tau0, float)

    def step(self, q_ref, qd_ref, qdd_ref, tau_ff_model, q, qd, j_tot, limiter):
        """limiter(tau_cmd, tau_prev) -> (tau_out, reason, lo, hi)."""
        e = q_ref - q
        ed = qd_ref - qd
        kd = np.maximum(self.kd, 2 * self.zeta * np.sqrt(self.kp * j_tot))
        ff = tau_ff_model + N2JM * qdd_ref + 0.8 * friction(qd_ref)
        soft = (CTRL.soft_limit_k * np.maximum(ROM[:, 0] - q, 0.0)
                - CTRL.soft_limit_k * np.maximum(q - ROM[:, 1], 0.0))
        tau_cmd = ff + self.kp * e + kd * ed + self.integ + soft
        tau_out, reason, lo, hi = limiter(tau_cmd, self.tau_prev)
        d_int = self.dt * self.ki * e
        if self.antiwindup:
            excess = tau_out - tau_cmd                     # < 0: clipped from above
            d_int = np.where((excess < -1e-9) & (d_int > 0) | (excess > 1e-9) & (d_int < 0), 0.0, d_int)
            i_new = self.integ + d_int
            back = self.dt * excess / CTRL.anti_windup_tt
            toward_zero = np.sign(back) == -np.sign(i_new)
            bled = np.where(i_new > 0, np.maximum(i_new + back, 0.0), np.minimum(i_new + back, 0.0))
            self.integ = np.where(toward_zero, bled, i_new)
        else:
            self.integ = self.integ + d_int
        self.tau_prev = tau_out
        return dict(tau_cmd=tau_cmd, tau_out=tau_out, reason=reason, lo=lo, hi=hi,
                    e=e, integ=self.integ.copy(), ff=ff)


class JointPlant:
    """Linearised error dynamics about the planned motion:
    J_tot e_dd = tau_out - tau_ID(ref) - N^2 J_m qdd_ref - friction(qd)."""

    def __init__(self, dt):
        self.dt = dt
        self.e = np.zeros(4)
        self.ed = np.zeros(4)
        self.edd = np.zeros(4)

    def step(self, tau_out, tau_id, qd_ref, qdd_ref, j_seg):
        j_tot = j_seg + N2JM
        qd = qd_ref + self.ed
        tau_joint = tau_out - N2JM * (qdd_ref + self.edd) - friction(qd)   # what the torque sensor sees
        self.edd = (tau_out - tau_id - N2JM * qdd_ref - friction(qd)) / j_tot
        self.ed += self.edd * self.dt
        self.e += self.ed * self.dt
        return tau_joint
