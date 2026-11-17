"""Play a unitree_rl_mjlab policy (policy.onnx + deploy.yaml) in Isaac Lab.

Cross-sim check for the G1 23-DoF: the deploy contract only, like g1_ctrl's
Velocity state and unitree_rl_mjlab's vel_eval.py (MuJoCo). Isaac Lab is used
for the physics of the Unitree-G1-23dof-Velocity-Flat task (same gains,
armature and action scale as mjlab); observations are built from deploy.yaml,
not from the task's observation manager. Events, terminations and curricula
are switched off, the robot starts in the default pose and each command case
runs settle -> walk, measuring the last seconds.

The ONNX graph (normalizer + MLP) is evaluated with torch, so onnxruntime is
not needed in this environment.

  python scripts/play_deploy_onnx.py \
    --policy-dir ~/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/velocity/v0
  # Newton (MJWarp) instead of PhysX: append presets=newton
  # Newton viewer: append --visualizer newton (needs newton 1.2.0 + mujoco 3.8.1 in env_isaaclab);
  # the kit viewer and video capture need a newer Isaac Sim than 6.0.0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

DEFAULT_CASES = "0.4,0,0;0.8,0,0;1.0,0,0;0,0,0.5;0,0,-0.5;0,0,1.0;0.4,0,0.5;0,0.3,0"

parser = argparse.ArgumentParser(description="Play a mjlab policy.onnx + deploy.yaml in Isaac Lab.")
parser.add_argument("--policy-dir", required=True, help="folder with exported/policy.onnx and params/deploy.yaml")
parser.add_argument("--onnx", default=None, help="override exported/policy.onnx")
parser.add_argument("--task", default="Unitree-G1-23dof-Velocity-Flat")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--cases", default=DEFAULT_CASES, help='"vx,vy,wz;..."')
parser.add_argument("--settle", type=float, default=2.0)
parser.add_argument("--walk", type=float, default=12.0)
parser.add_argument("--measure", type=float, default=8.0)
parser.add_argument("--no-smooth", action="store_true", help="step the command instead of g1_ctrl-like smoothing")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer="none")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import torch
import yaml
from onnx import numpy_helper

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401

FALL_TILT = 0.85  # rad, same as vel_eval.py


class OnnxMlp:
  """Evaluates the exported actor (Sub/Div normalizer, Gemm, Elu, ...) with torch."""

  def __init__(self, path: Path, device: str):
    model = onnx.load(str(path))
    self.graph = model.graph
    self.consts = {i.name: torch.tensor(numpy_helper.to_array(i), device=device) for i in self.graph.initializer}
    self.input = self.graph.input[0].name
    self.output = self.graph.output[0].name
    self.meta = {p.key: p.value for p in model.metadata_props}

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    v = dict(self.consts)
    v[self.input] = x
    for n in self.graph.node:
      a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
      i = [v[k] for k in n.input if k]
      op = n.op_type
      if op == "Gemm":
        A = i[0].T if a.get("transA", 0) else i[0]
        B = i[1].T if a.get("transB", 0) else i[1]
        y = a.get("alpha", 1.0) * A @ B
        if len(i) > 2:
          y = y + a.get("beta", 1.0) * i[2]
      elif op in ("Add", "Sub", "Mul", "Div"):
        y = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div}[op](i[0], i[1])
      elif op == "MatMul":
        y = i[0] @ i[1]
      elif op == "Elu":
        y = torch.nn.functional.elu(i[0], alpha=a.get("alpha", 1.0))
      elif op == "Relu":
        y = torch.relu(i[0])
      elif op == "Tanh":
        y = torch.tanh(i[0])
      elif op == "Identity":
        y = i[0]
      else:
        raise NotImplementedError(f"ONNX op {op}")
      v[n.output[0]] = y
    return v[self.output]


def switch_off_randomness(env_cfg) -> None:
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = 1.0e6
  for group in ("events", "terminations", "curriculum", "rewards"):
    cfg = getattr(env_cfg, group)
    for name in list(vars(cfg)):
      if not name.startswith("_"):
        setattr(cfg, name, None)
  # Only the rewards read it, and Newton 1.0.0 rejects the sensor arguments of this Isaac Lab.
  env_cfg.scene.contact_forces = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
  policy_dir = Path(args_cli.policy_dir).expanduser()
  dep = yaml.safe_load(open(policy_dir / "params/deploy.yaml"))
  onnx_path = Path(args_cli.onnx).expanduser() if args_cli.onnx else policy_dir / "exported/policy.onnx"

  switch_off_randomness(env_cfg)
  cases = args_cli.cases.split(";")
  env = gym.make(args_cli.task, cfg=env_cfg)
  uenv = env.unwrapped
  device = uenv.device
  policy = OnnxMlp(onnx_path, device)
  names = policy.meta["joint_names"].split(",")

  # Policy order -> action term order, and checks against deploy.yaml.
  term = uenv.action_manager.get_term("JointPositionAction")
  term_names = list(term._joint_names)
  perm = torch.tensor([term_names.index(n) for n in names], device=device)
  default = torch.tensor(dep["default_joint_pos"], device=device, dtype=torch.float32)
  scale = torch.tensor(dep["actions"]["JointPositionAction"]["scale"], device=device, dtype=torch.float32)
  t_scale = term._scale[0, perm] if torch.is_tensor(term._scale) else torch.full_like(scale, term._scale)
  t_offset = term._offset[0, perm] if torch.is_tensor(term._offset) else torch.full_like(default, term._offset)
  robot = uenv.scene["robot"]
  joint_cfg = SceneEntityCfg("robot", joint_names=names, preserve_order=True)
  joint_cfg.resolve(uenv.scene)
  print(f"[deploy] {onnx_path}")
  print(f"[deploy] action scale max diff vs deploy.yaml {float((t_scale - scale).abs().max()):.4f}, "
        f"default pose max diff {float((t_offset - default).abs().max()):.4f}")
  stiff = torch.as_tensor(robot.data.joint_stiffness if hasattr(robot.data, "joint_stiffness") else
                          robot.data.default_joint_stiffness, device=device)[0, joint_cfg.joint_ids]
  print(f"[deploy] kp max diff vs deploy.yaml {float((stiff - torch.tensor(dep['stiffness'], device=device)).abs().max()):.3f}")
  print(f"[deploy] step_dt {uenv.step_dt} (deploy {dep['step_dt']})")

  obs_cfg = dep["observations"]
  terms = list(obs_cfg)
  hist = {k: obs_cfg[k].get("history_length", 1) or 1 for k in terms}
  oscale = {k: torch.tensor(obs_cfg[k]["scale"], device=device) if obs_cfg[k].get("scale") else None for k in terms}
  period = obs_cfg["gait_phase"]["params"]["period"]
  dt = dep["step_dt"]

  def frame(cmd, phase, last):
    if float(torch.linalg.norm(cmd)) >= 0.1:
      ph = torch.tensor([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], device=device)
    else:
      ph = torch.zeros(2, device=device)
    raw = {
      "base_ang_vel": mdp.base_ang_vel(uenv)[0],
      "projected_gravity": mdp.projected_gravity(uenv)[0],
      "velocity_commands": cmd,
      "gait_phase": ph,
      "joint_pos_rel": mdp.joint_pos(uenv, joint_cfg)[0] - default,
      "joint_vel_rel": mdp.joint_vel(uenv, joint_cfg)[0],
      "last_action": last,
    }
    return {k: raw[k] * oscale[k] if oscale[k] is not None else raw[k] for k in terms}

  results = []
  for case in cases:
    target_cmd = torch.tensor([float(x) for x in case.split(",")], device=device)
    # Steps run in inference mode, so the reset that touches their buffers must too.
    with torch.inference_mode():
      env.reset()
    cmd = torch.zeros(3, device=device)
    last = torch.zeros(len(names), device=device)
    phase = 0.0
    first = frame(cmd, phase, last)
    buf = {k: deque([first[k]] * hist[k], maxlen=hist[k]) for k in terms}
    n_settle = int(args_cli.settle / dt)
    n_total = n_settle + int(args_cli.walk / dt)
    n_meas0 = n_total - int(args_cli.measure / dt)
    vx, vy, wz, tilt_max, fell = [], [], [], 0.0, False
    with torch.inference_mode():
      for k in range(n_total):
        tgt = target_cmd if k >= n_settle else torch.zeros(3, device=device)
        cmd = tgt.clone() if args_cli.no_smooth else cmd + (tgt - cmd) * 0.15
        phase = (phase + dt / period) % 1.0
        f = frame(cmd, phase, last)
        for t in terms:
          buf[t].append(f[t])
        obs = torch.cat([torch.cat(list(buf[t])) for t in terms]).float()[None]
        act = policy(obs)[0]
        last = act.clone()
        env_act = torch.zeros(1, len(term_names), device=device)
        env_act[0, perm] = act
        env.step(env_act)
        g = mdp.projected_gravity(uenv)[0]
        tilt = float(torch.arccos(torch.clamp(-g[2], -1.0, 1.0)))
        tilt_max = max(tilt_max, tilt)
        if tilt > FALL_TILT:
          fell = True
          break
        if k >= n_meas0:
          lv = mdp.base_lin_vel(uenv)[0]
          vx.append(float(lv[0]))
          vy.append(float(lv[1]))
          wz.append(float(mdp.base_ang_vel(uenv)[0][2]))
        if not simulation_app.is_running():
          break

    def stat(x):
      return round(float(np.mean(x)), 3) if x else None

    r = dict(cmd=[float(x) for x in target_cmd], fell=fell, tilt_max_deg=round(np.degrees(tilt_max), 1),
             vx=stat(vx), vy=stat(vy), wz=stat(wz))
    results.append(r)
    print(json.dumps(r))
  env.close()


if __name__ == "__main__":
  main()
  simulation_app.close()
