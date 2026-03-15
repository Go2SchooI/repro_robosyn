"""
S1 验证/评估：在指定数据上算当前步与下一步 privileged target 的 MSE/MAE 及各维误差。

用法:
  # 验证某次训练得到的权重（权重在 log_dir 下按时间的子目录里）
  python -m belief_s1.eval_s1 --data_dir data/s1_baoding --checkpoint runs/belief_s1/20250315_143022/s1_last.pt

  # 只在 mujoco 数据上验证
  python -m belief_s1.eval_s1 --data_dir data/s1_baoding --checkpoint runs/belief_s1/YYYYMMDD_HHMMSS/s1_last.pt --domain mujoco

  # 用验证集目录（需事先预留一部分数据到另一目录）
  python -m belief_s1.eval_s1 --data_dir data/s1_baoding_val --checkpoint runs/belief_s1/YYYYMMDD_HHMMSS/s1_last.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.dataset import BeliefS1Dataset
from belief_s1.models import S1BeliefModel
from belief_s1.schema import N_DEPLOY, N_PRIV_TARGET

PRIV_DIM_NAMES = ("sin_phase", "cos_phase", "pair_xy_distance", "z_mean", "z_diff")


def parse_args():
    p = argparse.ArgumentParser(description="S1 验证/评估")
    p.add_argument("--data_dir", type=str, default="data/s1_baoding", help="数据目录（可用与训练不同的目录做验证集）")
    p.add_argument("--checkpoint", type=str, required=True, help="s1_last.pt 或 s1_epN.pt，路径通常为 runs/belief_s1/YYYYMMDD_HHMMSS/s1_last.pt")
    p.add_argument("--domain", type=str, default=None, help="仅评估该域 isaac|mujoco，不设则用全部")
    p.add_argument("--max_samples", type=int, default=None, help="最多评估多少条，不设则用全部")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = BeliefS1Dataset(
        data_dir=args.data_dir,
        domain=args.domain,
        max_samples=args.max_samples,
        load_pt=False,
    )
    if len(dataset) == 0:
        print("[eval_s1] 无数据，请检查 --data_dir（及 --domain）")
        return

    model = S1BeliefModel(
        obs_dim=N_DEPLOY,
        action_dim=22,
        priv_target_dim=N_PRIV_TARGET,
        latent_dim=128,
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    else:
        model.load_state_dict(ck)
    model.eval()

    errors_state = []
    errors_dyn = []
    n_dyn = 0  # 非 done 的条数，与训练时 L_dyn 一致

    with torch.no_grad():
        for i in range(len(dataset)):
            obs, action, priv_t, priv_t1, done = dataset[i]
            obs = obs.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            _, x_hat_t, x_hat_t1 = model(obs, action)
            e_state = (x_hat_t.cpu().numpy() - priv_t.numpy()) ** 2
            e_dyn = (x_hat_t1.cpu().numpy() - priv_t1.numpy()) ** 2
            errors_state.append(e_state)
            errors_dyn.append(e_dyn)
            if done.item() < 0.5:
                n_dyn += 1

    errors_state = np.concatenate(errors_state, axis=0)  # (N, 5)
    errors_dyn = np.concatenate(errors_dyn, axis=0)

    mse_state = np.mean(errors_state)
    mse_dyn = np.mean(errors_dyn)
    mae_state = np.mean(np.sqrt(errors_state))
    mae_dyn = np.mean(np.sqrt(errors_dyn))

    mse_state_per_dim = np.mean(errors_state, axis=0)
    mse_dyn_per_dim = np.mean(errors_dyn, axis=0)

    print("=== S1 验证 ===")
    print("checkpoint: {}".format(args.checkpoint))
    print("data_dir: {}  domain: {}  样本数: {}".format(args.data_dir, args.domain or "all", len(dataset)))
    print("当前步 target  MSE: {:.6f}  MAE: {:.6f}".format(mse_state, mae_state))
    print("下一步 target  MSE: {:.6f}  MAE: {:.6f}  (非 done 条数: {})".format(mse_dyn, mae_dyn, n_dyn))
    print("\n各维 MSE (当前步): {}".format(dict(zip(PRIV_DIM_NAMES, [round(x, 6) for x in mse_state_per_dim.tolist()]))))
    print("各维 MSE (下一步): {}".format(dict(zip(PRIV_DIM_NAMES, [round(x, 6) for x in mse_dyn_per_dim.tolist()]))))


if __name__ == "__main__":
    main()
