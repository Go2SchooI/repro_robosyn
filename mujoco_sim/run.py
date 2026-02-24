"""M5: Sim2Sim integration — run a trained Isaac Gym policy in MuJoCo.

Usage:
    python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml \\
                             --checkpoint runs/baoding/nn/last.pth \\
                             [--no-render] [--steps 3000]
"""

import argparse
import time

import numpy as np
import torch

from mujoco_sim.env import BaodingMujocoEnv
from mujoco_sim.policy import PolicyNetwork


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
    parser = argparse.ArgumentParser(description="Sim2Sim: Isaac Gym policy in MuJoCo")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo XML scene")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--no-render", action="store_true", help="Disable viewer")
    parser.add_argument("--steps", type=int, default=3000, help="Max inference steps")
    parser.add_argument("--slow", type=float, default=1.0, help="Playback slowdown factor (e.g. 5 = 5x slower)")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

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
