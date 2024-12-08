"""
Difference setup
python sim/play.py --task mini_ppo --sim_device cpu
python sim/sim2sim.py --load_model examples/standing_pro.pt --embodiment stompypro
python sim/sim2sim.py --load_model examples/standing_micro.pt --embodiment stompymicro
"""

import math
import os
import types
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Deque, Union

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.scripts.create_mjcf import load_embodiment
from sim.utils.args_parsing import parse_args_with_extras
from sim.utils.cmd_manager import CommandManager

import torch  # isort: skip


class Sim2simCfg:
    """Configuration for sim2sim transfer"""

    def __init__(
        self,
        embodiment: str,
        frame_stack: int = 15,
        c_frame_stack: int = 3,
        sim_duration: float = 60.0,
        dt: float = 0.001,
        decimation: int = 10,
        cycle_time: float = 0.4,
        tau_factor: float = 3,
        lin_vel: float = 2.0,
        ang_vel: float = 1.0,
        dof_pos: float = 1.0,
        dof_vel: float = 0.05,
        clip_observations: float = 18.0,
        clip_actions: float = 18.0,
        action_scale: float = 0.25,
    ):
        self.embodiment = embodiment
        self.robot = load_embodiment(embodiment)

        self.num_actions = len(self.robot.all_joints())

        self.frame_stack = frame_stack
        self.c_frame_stack = c_frame_stack
        self.num_single_obs = 11 + self.num_actions * self.c_frame_stack
        self.num_observations = int(self.frame_stack * self.num_single_obs)

        self.sim_duration = sim_duration
        self.dt = dt
        self.decimation = decimation

        self.cycle_time = cycle_time

        self.tau_factor = tau_factor
        self.tau_limit = (
            np.array(list(self.robot.effort().values()) + list(self.robot.effort().values())) * self.tau_factor
        )
        self.kps = np.array(list(self.robot.stiffness().values()) + list(self.robot.stiffness().values()))
        self.kds = np.array(list(self.robot.damping().values()) + list(self.robot.damping().values()))

        self.obs_scales = types.SimpleNamespace()
        self.obs_scales.lin_vel = lin_vel
        self.obs_scales.ang_vel = ang_vel
        self.obs_scales.dof_pos = dof_pos
        self.obs_scales.dof_vel = dof_vel
        self.obs_scales.quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.clip_observations = clip_observations
        self.clip_actions = clip_actions

        self.action_scale = action_scale


@dataclass
class SimulationState:
    """Holds the current state of the simulation"""

    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    gravity_vec: np.ndarray


class MujocoSimulator:
    """Handles all Mujoco-specific simulation logic"""

    def __init__(self, model_path: str, cfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = cfg.dt
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self._initialize_state()

    def _initialize_state(self):
        """Initialize simulation state"""
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.default_qpos = deepcopy(self.model.keyframe("default").qpos)
        except:
            print("No default position found, using zero initialization")
            self.default_qpos = np.zeros_like(self.data.qpos)

        print("Default position:", self.default_qpos[-self.cfg.num_actions :])
        if self.default_qpos is not None:
            print("Loaded joint positions:")
            for i, joint_name in enumerate(self.cfg.robot.all_joints()):
                print(f"{joint_name}: {self.default_qpos[-len(self.cfg.robot.all_joints()) + i]}")

        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        mujoco.mj_step(self.model, self.data)

        # Print joint information
        for ii in range(len(self.data.ctrl) + 1):
            print(self.data.joint(ii).id, self.data.joint(ii).name)

    def get_state(self) -> SimulationState:
        """Extract current simulation state"""
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        r = R.from_quat(quat)
        base_lin_vel = r.apply(self.data.qvel[:3], inverse=True)
        base_ang_vel = self.data.sensor("angular-velocity").data
        gravity_vec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        return SimulationState(
            q=self.data.qpos.astype(np.double),
            dq=self.data.qvel.astype(np.double),
            quat=quat.astype(np.double),
            base_lin_vel=base_lin_vel.astype(np.double),
            base_ang_vel=base_ang_vel.astype(np.double),
            gravity_vec=gravity_vec.astype(np.double),
        )

    def step(self, tau: np.ndarray):
        """Step the simulation forward"""
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def close(self):
        """Clean up resources"""
        self.viewer.close()


class PolicyWrapper:
    """Handles policy loading and inference"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.policy = self._load_policy()

    def _load_policy(self) -> Union[torch.jit._script.RecursiveScriptModule, Any]:
        if "pt" in self.model_path:
            return torch.jit.load(self.model_path)
        elif "onnx" in self.model_path:
            import onnxruntime as ort

            return ort.InferenceSession(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        if isinstance(self.policy, torch.jit._script.RecursiveScriptModule):
            return self.policy(torch.tensor(input_data))[0].detach().numpy()
        else:
            ort_inputs = {self.policy.get_inputs()[0].name: input_data}
            return self.policy.run(None, ort_inputs)[0][0]


class Controller:
    """Handles control logic and observation processing"""

    def __init__(self, cfg, policy_wrapper: PolicyWrapper):
        self.cfg = cfg
        self.policy = policy_wrapper

        self.obs_history: Deque = deque(maxlen=cfg.frame_stack)
        self.last_actions = np.zeros(cfg.num_actions, dtype=np.float32)
        self.last_last_actions = np.zeros(cfg.num_actions, dtype=np.float32)
        self.action = np.zeros(cfg.num_actions, dtype=np.float32)

        # Initialize observation history
        for _ in range(cfg.frame_stack):
            self.obs_history.append(np.zeros([1, cfg.num_single_obs], dtype=np.float32))
        self.default_dof_pos = np.array(
            [self.cfg.robot.default_standing()[joint_name] for joint_name in self.cfg.robot.all_joints()]
        )

    def _process_observation(self, state: SimulationState, count: int) -> np.ndarray:
        """Process raw state into policy observation"""
        obs = np.zeros([1, self.cfg.num_single_obs], dtype=np.float32)

        phase = count * self.cfg.dt / self.cfg.cycle_time

        # Commands
        commands = np.array(
            [
                self.x_vel_cmd * self.cfg.obs_scales.lin_vel,
                self.y_vel_cmd * self.cfg.obs_scales.lin_vel,
                self.yaw_vel_cmd * self.cfg.obs_scales.ang_vel,
            ]
        )

        # Extract joint states
        q = state.q[-self.cfg.num_actions :]
        dq = state.dq[-self.cfg.num_actions :]

        # Extract base states
        eu_ang = self._quat_to_euler(state.quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        if count % 1000 == 0:
            print("\nSim2sim Raw States:")
            print(f"Raw Base Lin Vel: {state.base_lin_vel}")
            print(f"Raw Base Ang Vel: {state.base_ang_vel}")
            print(f"Raw DOF Pos: {state.q[-self.cfg.num_actions:]}")
            print(f"Raw DOF Vel: {state.dq[-self.cfg.num_actions:]}")

            print("\nSim2sim Observation Components:")
            print(f"Phase: sin={obs[0,0]:.3f}, cos={obs[0,1]:.3f}")
            print(f"Commands: {obs[0,2:5]}")
            print(f"Joint Positions: {obs[0,5:self.cfg.num_actions+5]}")
            print(f"Joint Velocities: {obs[0,self.cfg.num_actions+5:2*self.cfg.num_actions+5]}")
            print(f"Actions: {obs[0,2*self.cfg.num_actions+5:3*self.cfg.num_actions+5]}")
            print(f"Base Angular Velocity: {obs[0,3*self.cfg.num_actions+5:3*self.cfg.num_actions+8]}")
            print(f"Euler Angles: {obs[0,3*self.cfg.num_actions+8:3*self.cfg.num_actions+11]}")

        obs[0, 0] = math.sin(2 * math.pi * phase)
        obs[0, 1] = math.cos(2 * math.pi * phase)
        obs[0, 2:5] = commands
        obs[0, 5 : (self.cfg.num_actions + 5)] = (q - self.default_dof_pos) * self.cfg.obs_scales.dof_pos
        obs[0, (self.cfg.num_actions + 5) : (2 * self.cfg.num_actions + 5)] = dq * self.cfg.obs_scales.dof_vel
        obs[0, (2 * self.cfg.num_actions + 5) : (3 * self.cfg.num_actions + 5)] = self.action
        obs[0, (3 * self.cfg.num_actions + 5) : (3 * self.cfg.num_actions + 5) + 3] = state.base_ang_vel
        obs[0, (3 * self.cfg.num_actions + 5) + 3 : (3 * self.cfg.num_actions + 5) + 2 * 3] = eu_ang

        return np.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def compute_action(self, state: SimulationState, count: int) -> np.ndarray:
        """Compute control action based on current state"""
        # Update observation history
        obs = self._process_observation(state, count)
        self.obs_history.append(obs)

        # Stack observations
        policy_input = np.zeros([1, self.cfg.num_observations], dtype=np.float32)
        for i in range(self.cfg.frame_stack):
            policy_input[0, i * self.cfg.num_single_obs : (i + 1) * self.cfg.num_single_obs] = self.obs_history[i][0, :]

        # Update action history
        self.last_last_actions = self.last_actions.copy()
        self.last_actions = self.action.copy()

        # Get policy output and scale
        self.action[:] = self.policy(policy_input)
        self.action = np.clip(self.action, -self.cfg.clip_actions, self.cfg.clip_actions)
        target_dof_pos = self.action * self.cfg.action_scale

        if count % 1000 == 0:
            print("\nAction Computation:")
            print(f"Raw Policy Output: {self.action}")
            print(f"Target DOF Positions: {target_dof_pos}")

        # Compute PD control
        q = state.q[-self.cfg.num_actions :]
        dq = state.dq[-self.cfg.num_actions :]
        target_dq = np.zeros(self.cfg.num_actions, dtype=np.double)

        # PD control with scaling
        tau = self._pd_control(
            target_dof_pos, q, self.cfg.kps, target_dq, dq, self.cfg.kds, self.cfg.robot.default_standing()
        )

        return np.clip(tau, -self.cfg.tau_limit, self.cfg.tau_limit)

    def _pd_control(self, target_q, q, kp, target_dq, dq, kd, default_dict):
        """PD control calculation"""
        # Convert default dictionary to array in correct order
        default = np.array([default_dict[joint_name] for joint_name in self.cfg.robot.all_joints()])
        return kp * (target_q + default - q) + kd * (target_dq - dq)


def run_simulation(cfg: Sim2simCfg, policy_path: str, command_mode: str = "fixed", legs_only: bool = False):
    """Main simulation loop"""
    # Initialize components
    model_dir = os.environ.get("MODEL_DIR") or "sim/resources"
    simulator = MujocoSimulator(
        model_path=f"{model_dir}/{cfg.embodiment}/robot" + ("_fixed" if legs_only else "") + ".xml", cfg=cfg
    )
    policy = PolicyWrapper(policy_path)
    controller = Controller(cfg, policy)
    cmd_manager = CommandManager(num_envs=1, mode=command_mode)

    # Main simulation loop
    for count in tqdm(range(int(cfg.sim_duration / cfg.dt)), desc="Simulating..."):
        commands = cmd_manager.update(cfg.dt)
        controller.x_vel_cmd = commands[0, 0].item()
        controller.y_vel_cmd = commands[0, 1].item()
        controller.yaw_vel_cmd = commands[0, 2].item()
        state = simulator.get_state()

        # Compute control at policy rate
        if count % cfg.decimation == 0:
            tau = controller.compute_action(state, count)
            simulator.step(tau)

    simulator.close()
    cmd_manager.close()


def add_sim2sim_arguments(parser):
    """Add sim2sim-specific arguments."""
    # Model loading
    parser.add_argument("--load_model", type=str, required=True, help="Path to model file")

    # Robot configuration
    parser.add_argument("--embodiment", type=str, required=True, help="Robot embodiment type")

    # Control
    parser.add_argument(
        "--command_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oscillating", "keyboard", "random"],
        help="Command mode for robot control",
    )


if __name__ == "__main__":
    args = parse_args_with_extras(add_sim2sim_arguments)
    print("Arguments:", vars(args))
    cfg = Sim2simCfg(
        args.embodiment,
        sim_duration=60.0,
        dt=0.001,
        decimation=20,
        cycle_time=0.5,
        tau_factor=4.0,
    )
    run_simulation(cfg, args.load_model, args.command_mode)