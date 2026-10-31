"""
Intent / safety supervisor (runs on the safety MCU at 1 kHz).

BCI intents only *request* a transition; nothing moves until the user confirms
with the crutch thumb button inside EEG_CFG.confirm_timeout_s and the
preconditions for that transition hold.  Hard faults latch FAULT_HOLD (brakes on
when standing); a low battery forces a guided stop/sit.
"""
import math

from params import ACT, BATT, CTRL, EEG_CFG

STATE_NAMES = ("SITTING", "STANDING_UP", "STANDING", "WALKING", "STOPPING", "SITTING_DOWN", "FAULT_HOLD")
SITTING, STANDING_UP, STANDING, WALKING, STOPPING, SITTING_DOWN, FAULT_HOLD = range(7)

ADVANCE = {SITTING: ("STAND", STANDING_UP), STANDING: ("WALK", WALKING)}     # FEET_MI
RETREAT = {WALKING: ("STOP", STOPPING), STANDING: ("SIT", SITTING_DOWN)}     # HANDS_MI
TRACK_FAULT_S = 0.05
THERMAL_FAULT_S = 1.0
HEARTBEAT_TIMEOUT_S = 0.5


class Supervisor:
    def __init__(self, body_weight_n):
        self.state = SITTING
        self.bw = body_weight_n
        self.pending = None
        self.log = []                 # (t, text)
        self.track_timer = 0.0
        self.therm_timer = 0.0
        self.fault = None

    def _say(self, t, text):
        self.log.append((round(t, 3), text))

    # -- BCI intent -------------------------------------------------------------
    def intent(self, t, cls, p):
        table = ADVANCE if cls == "FEET_MI" else RETREAT
        self._say(t, f"BCI: {cls} detected (p={p:.2f})")
        if self.state not in table:
            self._say(t, f"{cls} ignored in {STATE_NAMES[self.state]}")
            return None
        action, nxt = table[self.state]
        self.pending = dict(action=action, next=nxt, deadline=t + EEG_CFG.confirm_timeout_s)
        self._say(t, f"Request {action} - press crutch button to confirm")
        return action

    def preconditions(self, action, sig):
        w = self.bw
        if sig["v_bus"] < BATT.uvlo_warn_v and action in ("STAND", "WALK"):
            return False, "battery low"
        if action == "STAND":
            if not (sig["contact_L"] and sig["contact_R"]):
                return False, "feet not on the floor"
        elif action == "WALK":
            if min(sig["grf_L"], sig["grf_R"]) < 0.2 * w:
                return False, "weight not on both feet"
            if sig["crutch_load"] < 0.05 * w:
                return False, "crutches not loaded"
        elif action == "SIT":
            if not sig["chair_behind"]:
                return False, "no seat detected behind"
        return True, ""

    # -- crutch thumb button --------------------------------------------------
    def button(self, t, sig):
        if self.pending is None:
            self._say(t, "Crutch button: nothing to confirm")
            return None
        if t > self.pending["deadline"]:
            self.pending = None
            return None
        p, self.pending = self.pending, None
        ok, why = self.preconditions(p["action"], sig)
        if not ok:
            self._say(t, f"{p['action']} blocked: {why}")
            return None
        self.state = p["next"]
        self._say(t, f"Crutch button confirmed -> {p['action']}")
        return p["action"]

    def motion_done(self, t):
        done = {STANDING_UP: (STANDING, "Standing"), WALKING: (STANDING, "Standing"),
                STOPPING: (STANDING, "Stopped - standing"), SITTING_DOWN: (SITTING, "Seated")}
        if self.state in done:
            self.state, text = done[self.state]
            self._say(t, text)

    # -- monitors ---------------------------------------------------------------
    def tick(self, t, dt, sig):
        """Returns 'FAULT', 'STOP', 'SIT' or None."""
        if self.pending and t > self.pending["deadline"]:
            self._say(t, f"Request {self.pending['action']} expired (not confirmed)")
            self.pending = None
        if self.state == FAULT_HOLD:
            return None
        if max(abs(x) for x in sig["track_err"]) > CTRL.max_tracking_error:
            self.track_timer += dt
        else:
            self.track_timer = 0.0
        if max(abs(x) for x in sig["tau_joint"]) > ACT.torque_sensor_sat_frac * ACT.torque_sensor_fs:
            return self._trip(t, "torque sensor saturated")
        if self.track_timer > TRACK_FAULT_S:
            return self._trip(t, f"tracking error > {math.degrees(CTRL.max_tracking_error):.0f} deg")
        if sig["v_bus"] > BATT.chopper_on_v + 2.0:
            return self._trip(t, "bus over-voltage")
        self.therm_timer = self.therm_timer + dt if max(sig["t_wind"]) >= ACT.t_max else 0.0
        if self.therm_timer > THERMAL_FAULT_S:
            return self._trip(t, "winding over-temperature")
        if sig["v_bus"] < BATT.uvlo_sit_v:
            if self.state == WALKING:
                self.state = STOPPING
                self._say(t, "Battery critical -> stopping")
                return "STOP"
            if self.state == STANDING and sig["chair_behind"]:
                self.state = SITTING_DOWN
                self._say(t, "Battery critical -> guided sit")
                return "SIT"
        if sig["heartbeat_age"] > HEARTBEAT_TIMEOUT_S:
            self.pending = None
            if self.state == WALKING:
                self.state = STOPPING
                self._say(t, "BCI link lost -> stopping")
                return "STOP"
        return None

    def _trip(self, t, why):
        self.state = FAULT_HOLD
        self.fault = why
        self._say(t, f"FAULT: {why} -> hold, brakes on")
        return "FAULT"
