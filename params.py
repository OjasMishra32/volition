"""
Single source of truth for the Volition lower-limb exoskeleton concept (SI units).

Every other module (CAD, sizing, control, BCI, simulation, video) imports from here,
so changing a number here changes the whole design consistently.

Coordinate conventions
----------------------
World frame: +X forward, +Y left, +Z up. Sagittal plane = XZ. Ground at Z = 0.

Absolute segment angles (radians in code):
  trunk_pitch  p    + = trunk leans forward (top of trunk moves +X)
  thigh, shank phi  + = distal end swings forward of the proximal joint
  foot         psi  + = toes up
Joint angles (what the actuators / encoders measure):
  hip   flexion      = phi_thigh + trunk_pitch
  knee  flexion      = phi_thigh - phi_shank
  ankle dorsiflexion = psi_foot - phi_shank

Rotation of a segment into the world (rotation about +Y by angle a = R_y(a)):
  trunk/pelvis : R_y(+p)      thigh/shank : R_y(-phi)      foot : R_y(-psi)

Segment-local CAD frames
------------------------
Each exported part has its origin on its proximal joint axis at Y = 0 (the body
midline), so world_point = [x_joint, 0, z_joint] + R(angle) @ local_point and the
part's Y coordinates are already its true lateral position.  Limb parts hang
along -Z, the foot points along +X, the pelvis/trunk frame points along +Z.
Parts are modelled for the LEFT side (Y > 0); right parts are mirrored in Y.
"""

from dataclasses import dataclass, field
import math

G = 9.81
PRODUCT = "Volition"          # product name used by the film, page and CAD exports


# --------------------------------------------------------------------------- user
@dataclass(frozen=True)
class User:
    """Design user for simulation. Exo adjusts for 1.55-1.95 m, up to 100 kg."""
    height: float = 1.80
    mass: float = 85.0
    # Winter (2009) segment fractions
    thigh_len_frac: float = 0.245
    shank_len_frac: float = 0.246
    foot_len_frac: float = 0.152
    ankle_height_frac: float = 0.039
    thigh_mass_frac: float = 0.100
    shank_mass_frac: float = 0.0465
    foot_mass_frac: float = 0.0145
    hat_mass_frac: float = 0.678          # head-arms-trunk
    thigh_com_frac: float = 0.433          # from proximal joint
    shank_com_frac: float = 0.433
    thigh_rgyr_frac: float = 0.323         # radius of gyration about COM / length
    shank_rgyr_frac: float = 0.302
    hat_com_above_hip: float = 0.33        # m, trunk upright
    hat_rgyr: float = 0.30                 # m, about COM (sagittal)
    hip_half_width: float = 0.09           # hip-joint-centre lateral offset
    max_mass: float = 100.0
    height_range: tuple = (1.55, 1.95)

    @property
    def thigh_len(self): return self.thigh_len_frac * self.height
    @property
    def shank_len(self): return self.shank_len_frac * self.height
    @property
    def foot_len(self): return self.foot_len_frac * self.height
    @property
    def ankle_height(self): return self.ankle_height_frac * self.height


# ------------------------------------------------------------------ exo structure
@dataclass(frozen=True)
class Structure:
    # lateral (Y) positions of joint hardware, left side (right = mirrored)
    hip_act_y: float = 0.216      # hip actuator mid-plane (inner face 0.185)
    knee_act_y: float = 0.181     # knee actuator mid-plane (inner face 0.150, leg surface ~0.14)
    ankle_hinge_y: float = 0.140  # passive ankle hinge (lateral stirrup)
    # adjustable link lengths (joint centre to joint centre)
    thigh_len_range: tuple = (0.38, 0.48)
    shank_len_range: tuple = (0.38, 0.48)
    adjust_step: float = 0.010    # detent pitch of telescoping links
    pelvis_width_range: tuple = (0.34, 0.44)  # between hip actuator inner faces
    # backpack (battery + electronics), in the pelvis/trunk frame
    backpack_size: tuple = (0.10, 0.30, 0.28)      # depth X, width Y, height Z
    backpack_center: tuple = (-0.205, 0.0, 0.17)
    # cuff positions as a fraction of link length from the proximal joint
    thigh_cuff_at: float = 0.60
    shank_cuff_at: float = 0.40
    # masses (kg), per side unless noted
    m_pelvis_frame: float = 2.6      # frame + belt + hip ab/ad hinges (total)
    m_backpack: float = 4.4          # battery 2.4 + electronics 1.2 + housing 0.8
    m_thigh_link: float = 0.85       # link + cuff, excl. actuator
    m_shank_link: float = 0.75
    m_foot: float = 0.65             # AFO footplate + stirrup + insole sensors
    # passive DOFs
    hip_abd_stiffness: float = 180.0      # N m/rad, centring spring on ab/adduction
    ankle_afo_stiffness: float = 120.0    # N m/rad about neutral (dorsi-assist AFO)
    ankle_afo_neutral: float = math.radians(2.0)
    ankle_rom: tuple = (math.radians(-15), math.radians(20))


# ---------------------------------------------------------------------- actuator
@dataclass(frozen=True)
class Actuator:
    """EXA-100: one actuator SKU used at both hips and both knees (4 per exo).

    Frameless BLDC + strain-wave gear + output torque sensor + dual encoders +
    spring-applied power-off brake.  Electrical constants use the amplitude-
    invariant dq convention: tau_m = Kt * i_q, back-EMF (phase peak) = (2/3) Kt w_m.
    Gear limits are for a size-25, 100:1 strain-wave class unit -- verify against
    the chosen vendor's datasheet before release.
    """
    name: str = "EXA-100"
    # motor
    pole_pairs: int = 10
    kt: float = 0.090             # N m / A (peak phase current)
    r_phase: float = 0.100        # ohm, phase-to-neutral
    l_phase: float = 85e-6        # H
    j_rotor: float = 3.0e-5       # kg m^2 incl. wave generator
    i_cont: float = 7.0           # A, continuous (winding 120 C, 40 C ambient)
    i_peak: float = 22.0          # A, drive peak current limit
    motor_max_speed: float = 6000 * 2 * math.pi / 60  # rad/s mechanical
    # thermal (winding -> ambient, lumped)
    r_th: float = 1.6             # K/W
    c_th: float = 180.0           # J/K  (tau ~ 290 s)
    t_ambient: float = 35.0       # C, inside a cover against the body
    t_derate_start: float = 100.0
    t_max: float = 120.0
    # gear
    ratio: float = 100.0
    eff_forward: float = 0.75     # motor drives load
    eff_backward: float = 0.60    # load drives motor (regeneration)
    gear_rated_torque: float = 51.0
    gear_repeated_peak: float = 108.0
    gear_momentary: float = 200.0
    coulomb_friction_out: float = 1.5    # N m at output
    viscous_out: float = 1.2             # N m s/rad at output
    # software limits at the output
    tau_peak_sw: float = 100.0
    tau_cont_sw: float = 45.0
    speed_limit_out: float = 4.0         # rad/s
    torque_slew_limit: float = 2500.0    # N m/s (protects gear & user)
    # sensing
    torque_sensor_fs: float = 150.0      # N m full-scale
    torque_sensor_sat_frac: float = 0.95
    motor_encoder_bits: int = 14
    output_encoder_bits: int = 19
    # brake
    brake_holding_motor: float = 1.5     # N m at motor shaft (=150 N m static at output)
    # drive electronics
    pwm_hz: float = 20_000
    current_loop_bw_hz: float = 1500
    svpwm_utilization: float = 1 / math.sqrt(3)   # max phase-peak voltage / Vbus
    # envelope
    housing_od: float = 0.100
    housing_len: float = 0.062
    mass: float = 1.9


# ------------------------------------------------------------------------ battery
@dataclass(frozen=True)
class Battery:
    """13S2P Li-ion (21700, 5 Ah cells) with BMS, hot-swappable."""
    series: int = 13
    parallel: int = 2
    cell_capacity_ah: float = 5.0
    cell_r_int: float = 0.020
    v_cell_max: float = 4.20
    v_cell_min: float = 3.00
    wiring_r: float = 0.020
    i_discharge_max: float = 40.0
    i_charge_max: float = 10.0          # limits regenerative current
    chopper_on_v: float = 56.0          # brake chopper threshold
    chopper_off_v: float = 55.0
    chopper_r: float = 4.0
    bus_capacitance: float = 2.0e-3     # F, sum of drive DC-link caps
    uvlo_warn_v: float = 42.0           # warn + derate
    uvlo_sit_v: float = 40.0            # force guided sit
    # cell open-circuit voltage vs state of charge (NMC, rough)
    ocv_soc: tuple = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00)
    ocv_v:   tuple = (3.00, 3.30, 3.45, 3.55, 3.68, 3.80, 3.97, 4.07, 4.20)

    @property
    def capacity_ah(self): return self.parallel * self.cell_capacity_ah
    @property
    def r_pack(self): return self.series * self.cell_r_int / self.parallel + self.wiring_r
    @property
    def v_nominal(self): return self.series * 3.6
    @property
    def energy_wh(self): return self.v_nominal * self.capacity_ah


# ----------------------------------------------------------------------------- EEG
@dataclass(frozen=True)
class EEG:
    fs: int = 500
    channels: tuple = ("FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                       "CP3", "CPz", "CP4", "Pz", "HEOG", "VEOG")
    adc: str = "2x ADS1299 (24-bit, daisy-chained), gain 24"
    adc_vref: float = 4.5
    adc_gain: int = 24
    clip_frac: float = 0.90             # flag samples beyond 90% of full-scale
    impedance_max_gel: float = 20e3
    impedance_max_dry_active: float = 500e3
    band: tuple = (8.0, 30.0)
    notch_hz: float = 60.0              # 50 Hz outside North America
    window_s: float = 1.0
    update_hz: float = 16.0
    classes: tuple = ("REST", "FEET_MI", "HANDS_MI")
    evidence_tau_s: float = 1.2         # leaky integrator
    evidence_threshold: float = 0.80
    min_dwell_s: float = 0.75
    refractory_s: float = 3.0
    confirm_timeout_s: float = 4.0      # crutch-button confirmation window

    @property
    def adc_full_scale(self): return self.adc_vref / self.adc_gain   # volts


# ------------------------------------------------------------------------ control
@dataclass(frozen=True)
class Control:
    current_loop_hz: int = 20_000
    joint_loop_hz: int = 1_000
    gait_loop_hz: int = 200
    supervisor_hz: int = 1_000
    # joint impedance/PID gains (output side), per joint type
    kp: dict = field(default_factory=lambda: {"hip": 2200.0, "knee": 2200.0})
    kd: dict = field(default_factory=lambda: {"hip": 95.0, "knee": 85.0})
    ki: dict = field(default_factory=lambda: {"hip": 5000.0, "knee": 5000.0})
    anti_windup_tt: float = 0.10        # back-calculation tracking time constant, s
    # feed-forward model uses an imperfect user mass to expose model error
    model_mass_error: float = -0.08     # model assumes 8 % lighter user
    # soft joint limits (virtual springs) and mechanical hard stops
    hip_rom: tuple = (math.radians(-20), math.radians(110))
    knee_rom: tuple = (math.radians(0), math.radians(110))
    soft_limit_margin: float = math.radians(5)
    soft_limit_k: float = 600.0         # N m/rad beyond the soft limit
    max_tracking_error: float = math.radians(12)   # supervisor fault threshold


# --------------------------------------------------------------------------- gait
@dataclass(frozen=True)
class Gait:
    step_length: float = 0.38
    step_time: float = 1.5
    double_support_frac: float = 0.25
    swing_clearance: float = 0.06
    stance_knee_target: float = math.radians(10)
    trunk_lean_walk: float = math.radians(6)
    crutch_support_walk: float = 0.25   # fraction of HAT vertical load via crutches
    # sit <-> stand
    seat_height: float = 0.48
    sts_duration: float = 3.0
    stand_to_sit_duration: float = 3.6
    sts_trunk_lean_peak: float = math.radians(38)
    crutch_support_sts: float = 0.30
    standing_knee: float = math.radians(4)


USER = User()
STRUCT = Structure()
ACT = Actuator()
BATT = Battery()
EEG_CFG = EEG()
CTRL = Control()
GAIT = Gait()

JOINTS = ("hip_L", "knee_L", "hip_R", "knee_R")   # powered joints (actuator order)
