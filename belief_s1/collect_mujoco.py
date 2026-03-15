"""
从 MuJoCo 环境采集 S1 格式数据：obs_deploy, action, priv_target_t, priv_target_t_plus_1, meta。

用法（在仓库根目录）:
  python -m belief_s1.collect_mujoco --xml mujoco_sim/assets/allegro_baoding.xml --out_dir data/s1_baoding --steps 5000
  python -m belief_s1.collect_mujoco --xml ... --checkpoint runs/baoding/nn/baoding.pth --action_noise 0.05
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.schema import (
    N_DEPLOY,
    N_PRIV_FULL,
    deploy_slice_from_full_obs,
    compute_priv_target_from_full_priv,
    get_meta_template,
)


def parse_args():
    p = argparse.ArgumentParser(description="MuJoCo S1 数据采集")
    p.add_argument("--xml", type=str, required=True, help="MuJoCo 场景 XML 路径")
    p.add_argument("--out_dir", type=str, default="data/s1_baoding", help="输出目录")
    p.add_argument("--steps", type=int, default=5000, help="总步数")
    p.add_argument("--checkpoint", type=str, default=None, help="策略 checkpoint，不传则用随机动作")
    p.add_argument("--action_noise", type=float, default=0.05, help="动作噪声标准差")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_render", action="store_true", help="不渲染")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from mujoco_sim.env import BaodingMujocoEnv
    policy = None
    if args.checkpoint:
        import torch
        from mujoco_sim.policy import PolicyNetwork
        policy = PolicyNetwork.from_checkpoint(args.checkpoint, device=args.device)
        policy.eval()

    env = BaodingMujocoEnv(xml_path=args.xml, render=not args.no_render)
    os.makedirs(args.out_dir, exist_ok=True)

    samples = []
    obs = env.reset()
    episode_id = 0
    step_in_ep = 0

    for step in range(args.steps):
        obs_deploy = deploy_slice_from_full_obs(obs)
        priv_full = obs[N_DEPLOY : N_DEPLOY + N_PRIV_FULL]
        priv_t = compute_priv_target_from_full_priv(priv_full)

        if policy is not None:
            import torch
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0).cpu().numpy()
        else:
            action = np.zeros(22, dtype=np.float32)
        if args.action_noise > 0:
            action = action + np.random.randn(22).astype(np.float32) * args.action_noise
        action = np.clip(action, -1.0, 1.0)

        next_obs = env.step(action)
        pd_targets = env.get_pd_targets_isaac_order()

        next_priv_full = next_obs[N_DEPLOY : N_DEPLOY + N_PRIV_FULL]
        priv_t_plus_1 = compute_priv_target_from_full_priv(next_priv_full)

        ball1_z = next_obs[N_DEPLOY + 2]
        ball2_z = next_obs[N_DEPLOY + 7 + 2]
        done = bool(ball1_z < 0.05 or ball2_z < 0.05)

        meta = get_meta_template("mujoco", episode_id, step_in_ep)
        meta["done"] = done
        samples.append({
            "obs_deploy": obs_deploy.copy(),
            "action": action.copy(),
            "priv_target_t": priv_t.copy(),
            "priv_target_t_plus_1": priv_t_plus_1.copy(),
            "pd_targets": pd_targets.copy(),
            "meta": meta,
        })

        obs = next_obs
        step_in_ep += 1

        if done:
            obs = env.reset()
            episode_id += 1
            step_in_ep = 0

        if (step + 1) % 1000 == 0:
            print(f"[MuJoCo] step {step+1}/{args.steps}  samples={len(samples)}")

    # 保存为单文件 .npz（数组形式）；pd_targets 为送入 PD 的目标（Isaac 顺序）
    obs_deploy = np.array([s["obs_deploy"] for s in samples], dtype=np.float32)
    action = np.array([s["action"] for s in samples], dtype=np.float32)
    priv_t = np.array([s["priv_target_t"] for s in samples], dtype=np.float32)
    priv_t1 = np.array([s["priv_target_t_plus_1"] for s in samples], dtype=np.float32)
    pd_targets = np.array([s["pd_targets"] for s in samples], dtype=np.float32)
    meta = np.array([s["meta"] for s in samples], dtype=object)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"mujoco_s1_{time_str}.npz"
    out_path = os.path.join(args.out_dir, out_name)
    np.savez_compressed(out_path, obs_deploy=obs_deploy, action=action, priv_target_t=priv_t, priv_target_t_plus_1=priv_t1, pd_targets=pd_targets, meta=meta)
    print(f"[MuJoCo] 已保存 {len(samples)} 条样本 -> {out_path}")
    env.close()


if __name__ == "__main__":
    main()
