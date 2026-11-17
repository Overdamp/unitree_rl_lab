"""G1 23-DoF velocity task.

Reuses the G1 29-DoF MDP (rewards, terminations, observations, terrain). All
joint and body terms there are regex based and match the 23-DoF model too.

Robot parameters (gains, armature, effort limits, home pose, action scale)
match unitree_rl_mjlab ``src/assets/robots/unitree_g1/g1_23dof_constants.py``
so that policies from both stacks share one deploy contract (deploy.yaml).
"""

import importlib

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG

# The 29-DoF package name starts with a digit, so a plain import statement is not possible.
_g1_29dof = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")


##
# Actuator parameters, same formulas as unitree_rl_mjlab.
##


def _reflected_inertia(rotor_inertia: tuple[float, float, float], gear_ratio: tuple[float, float, float]) -> float:
    """Reflected inertia of a two-stage planetary gearbox."""
    return (
        rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
        + rotor_inertia[1] * gear_ratio[2] ** 2
        + rotor_inertia[2]
    )


ARMATURE_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
ARMATURE_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
ARMATURE_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _actuator(joint_names_expr, armature, effort_limit, velocity_limit) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit_sim=effort_limit,
        velocity_limit_sim=velocity_limit,
        stiffness=armature * NATURAL_FREQ**2,
        damping=2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
        armature=armature,
    )


G1_23DOF_ACTUATORS = {
    "N5020": _actuator(
        [".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
         ".*_wrist_roll_joint"],
        ARMATURE_5020, 25.0, 37.0,
    ),
    "N7520-14": _actuator([".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"], ARMATURE_7520_14, 88.0, 32.0),
    "N7520-22": _actuator([".*_hip_roll_joint", ".*_knee_joint"], ARMATURE_7520_22, 139.0, 20.0),
    # Ankles are 4-bar linkages driven by two 5020 motors, nominal 1:1 ratio.
    "N5020-ankle": _actuator([".*_ankle_pitch_joint", ".*_ankle_roll_joint"], ARMATURE_5020 * 2, 50.0, 37.0),
}

# Same as unitree_rl_mjlab: 0.25 * effort_limit / stiffness per actuator group.
G1_23DOF_ACTION_SCALE = {
    expr: 0.25 * act.effort_limit_sim / act.stiffness
    for act in G1_23DOF_ACTUATORS.values()
    for expr in act.joint_names_expr
}

G1_23DOF_CFG = UNITREE_G1_23DOF_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=G1_23DOF_ACTUATORS,
)


def _apply_g1_23dof(cfg) -> None:
    cfg.scene.robot = G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.JointPositionAction.scale = G1_23DOF_ACTION_SCALE


@configclass
class RobotEnvCfg(_g1_29dof.RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_23dof(self)


@configclass
class RobotPlayEnvCfg(_g1_29dof.RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_23dof(self)


@configclass
class RobotFlatEnvCfg(_g1_29dof.RobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_23dof(self)


@configclass
class RobotFlatPlayEnvCfg(_g1_29dof.RobotFlatPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_g1_23dof(self)
