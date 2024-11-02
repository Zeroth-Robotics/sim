"""
python sim/macro.py  --embodiment stompymicro
"""
import argparse
import math
import os
from collections import deque
from copy import deepcopy

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.scripts.create_mjcf import load_embodiment


class Sim2simCfg:
    def __init__(
        self,
        embodiment,
        frame_stack=15,
        c_frame_stack=3,
        sim_duration=60.0,
        dt=0.001,
        decimation=10,
        cycle_time=0.4,
        tau_factor=3,
        lin_vel=2.0,
        ang_vel=1.0,
        dof_pos=1.0,
        dof_vel=0.05,
        clip_observations=18.0,
        clip_actions=18.0,
        action_scale=0.25,
    ):
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

        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel

        self.clip_observations = clip_observations
        self.clip_actions = clip_actions

        self.action_scale = action_scale


def pd_control(target_q, q, kp, target_dq, dq, kd, default):
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) - kd * dq


def get_scripted_joint_targets(current_time, cfg, joints):
    """
    Returns the target joint positions for the robot at the given time.

    Args:
        current_time (float): The current simulation time.
        cfg (Sim2simCfg): The simulation configuration.

    Returns:
        np.ndarray: The target joint positions.
    """
    global save_q
    # Total duration for the scripted behavior
    total_duration = 7.0  # seconds

    # Initialize target positions
    target_q = np.zeros(cfg.num_actions, dtype=np.double)

    # Define time intervals for different actions
    # You can adjust these intervals as needed
    t0 = .033  # Time to let it fall
    t1 = 1.0  # Time to move arms to position
    t2 = 2.0  # Time to bend knees
    t3 = 3.0  # Time to stand up

    # lie down
    # Lie down to arms up (0 to t1)
    # breakpoint()
    # let it fall

    if current_time <= t0:
        pass
    elif current_time <= t1:
        progress_1 = (current_time - t0) / (t1 - t0)
        # Interpolate arm joints from initial to target position
        # Assuming arm joints are indices 0 and 1 (adjust based on your robot)
        arm_joint_indices = [joints["right_shoulder_pitch"], joints["left_shoulder_pitch"]]  # Replace with actual indices
        arm_target_positions = np.deg2rad([-90, 90])  # Raise arms up

        for idx, joint in enumerate(arm_joint_indices):
            target_q[joint] = np.interp(progress_1, [0, 1], [0, arm_target_positions[idx]])

        save_q = target_q
    # Arms up to bend knees (t1 to t2)
    elif current_time <= t2:
        target_q = save_q
        progress = (current_time - t1) / (t2 - t1)

        knee_joint_indices = [joints["right_knee_pitch"], joints["left_knee_pitch"]]  # Replace with actual indices
        knee_bend_positions = np.deg2rad([-90, 90])  # Bend knees
        for idx, joint in enumerate(knee_joint_indices):
            target_q[joint] = np.interp(progress, [0, 1], [0, knee_bend_positions[idx]])
        save_q = target_q
    # Bend knees to stand up (t2 to t3)
    elif current_time <= t3:
        target_q = save_q
        progress = (current_time - t2) / (t3 - t2)

        # Knees extend to standing position
        pitch_joint_indices = [joints["right_hip_pitch"], joints["left_hip_pitch"]]  # Replace with actual indices
        pitch_positions = np.deg2rad([-90, 90])  # Bend knees
        for idx, joint in enumerate(pitch_joint_indices):
            target_q[joint] = np.interp(progress, [0, 1], [0, pitch_positions[idx]])
        save_q = target_q
    else:
        # After t3, maintain standing position
        # Arms remain up
        target_q = save_q
        arm_joint_indices = [0, 1]
        arm_target_positions = np.deg2rad([90, 90])
        for idx in arm_joint_indices:
            target_q[idx] = arm_target_positions[idx]
        # Knees fully extended
        knee_joint_indices = [2, 3]
        for idx in knee_joint_indices:
            target_q[idx] = 0.0

    return target_q

def run_mujoco_scripted(cfg):
    """
    Run the Mujoco simulation using scripted joint positions.

    Args:
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model_dir = os.environ.get("MODEL_DIR")
    mujoco_model_path = f"{model_dir}/{args.embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    # Initialize default positions
    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.num_actions:]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(cfg.num_actions)

    mujoco.mj_step(model, data)
    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    mujoco.mj_step(model, data)
    joints = {}
    for ii in range(1, len(data.ctrl)):
        joints[data.joint(ii).name] = data.joint(ii).id - 1

    # Initialize target joint positions
    target_q = np.zeros((cfg.num_actions), dtype=np.double)

    # Simulation loop
    sim_steps = int(cfg.sim_duration / cfg.dt)
    for step in tqdm(range(sim_steps), desc="Simulating..."):
        current_time = step * cfg.dt
        q = data.qpos[-cfg.num_actions:].astype(np.double)
        dq = data.qvel[-cfg.num_actions:].astype(np.double)

        # Update target_q based on scripted behavior
        target_q = get_scripted_joint_targets(current_time, cfg, joints)
        target_dq = np.zeros((cfg.num_actions), dtype=np.double)
        tau = pd_control(target_q, q, cfg.kps, target_dq, dq, cfg.kds, default)

        print("tau:", tau)
        print("q:", q)
        print("target_q:", target_q)
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()

    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="embodiment")
    args = parser.parse_args()


    if args.embodiment == "stompypro":
        cfg = Sim2simCfg(
            args.embodiment,
            sim_duration=60.0,
            dt=0.001,
            decimation=10,
            cycle_time=0.4,
            tau_factor=3.0,
        )
    elif args.embodiment == "stompymicro":
        cfg = Sim2simCfg(
            args.embodiment,
            sim_duration=60.0,
            dt=0.001,
            decimation=10,
            cycle_time=0.4,
            tau_factor=2.,
        )

    run_mujoco_scripted(cfg)
