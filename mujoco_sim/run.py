"""M5: Sim2Sim integration — run a trained Isaac Gym policy in MuJoCo.

Usage (policy inference):
    python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml \\
                             --checkpoint runs/baoding/nn/last.pth \\
                             [--no-render] [--steps 3000]

Usage (offline trajectory replay from Isaac Gym env0 CSV):
    python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml \\
                             --trajectory-csv /path/to/env0_trajectory.csv \\
                             [--no-render] [--slow 1.0]
"""

import argparse
import csv
import time

import numpy as np
import torch

from mujoco_sim.env import BaodingMujocoEnv
from mujoco_sim.policy import PolicyNetwork
from mujoco_sim.utils import ARM_JOINT_NAMES, HAND_JOINT_NAMES


def load_trajectory_csv(path: str) -> np.ndarray:
    """Load trajectory CSV and remap columns to MuJoCo joint order.

    Isaac Gym sorts branching URDF children lexicographically, so the hand DOF
    order is [finger0, thumb, finger1, finger2] rather than [finger0..3, thumb].
    The CSV header (written by the fixed Isaac recording code) contains the true
    DOF names, so we parse it to build a column permutation.

    Returns:
        (N, 22) array in MuJoCo order: [joint1..joint6, joint_0.0..joint_15.0].
    """
    mujoco_order = list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES)

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        csv_joint_names = header[1:]  # skip "step"

        name_to_csv_col = {name: i for i, name in enumerate(csv_joint_names)}
        permutation = []
        for name in mujoco_order:
            if name not in name_to_csv_col:
                raise ValueError(
                    f"Joint '{name}' not found in CSV header. "
                    f"Header joints: {csv_joint_names}"
                )
            permutation.append(name_to_csv_col[name])

        for row in reader:
            if len(row) < 23:
                continue
            vals = [float(row[i + 1]) for i in range(22)]
            reordered = [vals[permutation[i]] for i in range(22)]
            rows.append(reordered)

    traj = np.array(rows, dtype=np.float64)
    print(f"  CSV header joint order: {csv_joint_names}")
    print(f"  Remapped to MuJoCo order: {mujoco_order}")
    return traj


def run_offline_trajectory(
    xml_path: str,
    trajectory_path: str,
    render: bool = True,
    slow_factor: float = 1.0,
):
    """Run MuJoCo with joint targets from a CSV (no policy)."""
    print("=" * 60)
    print("  MuJoCo offline trajectory tracking (sim2sim)")
    print("=" * 60)

    print("\n[1/2] Loading trajectory CSV...")
    trajectory = load_trajectory_csv(trajectory_path)
    print(f"  Loaded {len(trajectory)} steps (22 DOF per step).")

    print("\n[2/2] Creating MuJoCo environment (Isaac-matched: 6 PD x 2 substeps)...")
    env = BaodingMujocoEnv(xml_path=xml_path, render=render)
    obs = env.reset()

    print("\nReplaying trajectory...")
    for step, targets in enumerate(trajectory):
        obs = env.step_from_targets(targets)

        ball1_pos = env.get_ball_pos(0)
        ball2_pos = env.get_ball_pos(1)
        hand_qpos = env.get_hand_qpos()

        if step % 100 == 0:
            print(
                f"  Step {step:5d}/{len(trajectory)} | "
                f"Ball1: [{ball1_pos[0]:.3f}, {ball1_pos[1]:.3f}, {ball1_pos[2]:.3f}] | "
                f"Ball2: [{ball2_pos[0]:.3f}, {ball2_pos[1]:.3f}, {ball2_pos[2]:.3f}] | "
                f"Hand qpos range: [{hand_qpos.min():.3f}, {hand_qpos.max():.3f}]"
            )

        if render:
            if env.viewer is not None and not env.viewer.is_running():
                print("  Viewer closed by user.")
                break
            time.sleep(max(0, env.dt * env.control_freq_inv * slow_factor - 0.001))

    env.close()
    print("\nDone.")


def run(
    xml_path: str,
    checkpoint_path: str,
    render: bool = True,
    max_steps: int = 3000,
    device: str = "cpu",
    slow_factor: float = 1.0,
):
    print("=" * 60)
    print("  Allegro Baoding Sim2Sim: Isaac Gym -> MuJoCo")
    print("=" * 60)

    print("\n[1/3] Loading policy...")
    policy = PolicyNetwork.from_checkpoint(checkpoint_path, device=device)

    print("\n[2/3] Creating MuJoCo environment...")
    env = BaodingMujocoEnv(xml_path=xml_path, render=render)

    print("\n[3/3] Running inference loop...")
    obs = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(obs_tensor).squeeze(0).cpu().numpy()

        obs = env.step(action)

        ball1_pos = env.get_ball_pos(0)
        ball2_pos = env.get_ball_pos(1)
        hand_qpos = env.get_hand_qpos()

        if step % 100 == 0:
            print(
                f"  Step {step:5d} | "
                f"Ball1: [{ball1_pos[0]:.3f}, {ball1_pos[1]:.3f}, {ball1_pos[2]:.3f}] | "
                f"Ball2: [{ball2_pos[0]:.3f}, {ball2_pos[1]:.3f}, {ball2_pos[2]:.3f}] | "
                f"Hand qpos range: [{hand_qpos.min():.3f}, {hand_qpos.max():.3f}]"
            )

        if ball1_pos[2] < 0.05 and ball2_pos[2] < 0.05:
            print(f"  Both balls dropped at step {step}. Resetting...")
            obs = env.reset()

        if render:
            if env.viewer is not None and not env.viewer.is_running():
                print("  Viewer closed by user.")
                break
            time.sleep(max(0, env.dt * env.control_freq_inv * slow_factor - 0.001))

    env.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim: Isaac Gym policy in MuJoCo or offline trajectory replay")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo XML scene")
    parser.add_argument("--checkpoint", help="Path to .pth checkpoint (omit when using --trajectory-csv)")
    parser.add_argument("--trajectory-csv", help="Path to CSV of env0 joint targets from Isaac Gym (offline replay)")
    parser.add_argument("--no-render", action="store_true", help="Disable viewer")
    parser.add_argument("--steps", type=int, default=3000, help="Max inference steps (policy mode only)")
    parser.add_argument("--slow", type=float, default=1.0, help="Playback slowdown factor (e.g. 5 = 5x slower)")
    parser.add_argument("--device", default="cpu", help="Torch device (policy mode only)")
    args = parser.parse_args()

    if args.trajectory_csv:
        if args.checkpoint:
            parser.error("Do not pass --checkpoint when using --trajectory-csv")
        run_offline_trajectory(
            xml_path=args.xml,
            trajectory_path=args.trajectory_csv,
            render=not args.no_render,
            slow_factor=args.slow,
        )
    else:
        if not args.checkpoint:
            parser.error("Either --checkpoint or --trajectory-csv is required")
        run(
            xml_path=args.xml,
            checkpoint_path=args.checkpoint,
            render=not args.no_render,
            max_steps=args.steps,
            device=args.device,
            slow_factor=args.slow,
        )


if __name__ == "__main__":
    main()
