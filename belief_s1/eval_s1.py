"""
S1 评估：当前/下一步 privileged target 预测误差，按 meta 分层（phase/success/segment）。

用法:
  python -m belief_s1.eval_s1 --data_dir data/s1_baoding --checkpoint runs/belief_s1/s1_last.pt [--domain isaac]
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.dataset import BeliefS1Dataset
from belief_s1.models import S1BeliefModel
from belief_s1.schema import N_PRIV_TARGET


def parse_args():
    p = argparse.ArgumentParser(description="S1 评估")
    p.add_argument("--data_dir", type=str, default="data/s1_baoding")
    p.add_argument("--checkpoint", type=str, required=True, help="s1_last.pt 或 s1_ep*.pt")
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = BeliefS1Dataset(data_dir=args.data_dir, domain=args.domain, max_samples=args.max_samples)
    if len(dataset) == 0:
        print("无数据")
        return
    model = S1BeliefModel(priv_target_dim=N_PRIV_TARGET).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    else:
        model.load_state_dict(ck)
    model.eval()

    errors_state = []
    errors_dyn = []
    by_domain = defaultdict(lambda: {"state": [], "dyn": []})
    by_success = defaultdict(lambda: {"state": [], "dyn": []})

    with torch.no_grad():
        for i in range(len(dataset)):
            obs, action, priv_t, priv_t1 = dataset[i]
            obs = obs.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            _, x_hat_t, x_hat_t1 = model(obs, action)
            e_state = (x_hat_t.cpu().numpy() - priv_t.numpy()) ** 2
            e_dyn = (x_hat_t1.cpu().numpy() - priv_t1.numpy()) ** 2
            errors_state.append(e_state)
            errors_dyn.append(e_dyn)
            meta = dataset.samples[i].get("meta", {})
            dn = meta.get("domain_name", "unknown")
            by_domain[dn]["state"].append(e_state)
            by_domain[dn]["dyn"].append(e_dyn)
            succ = meta.get("success", -1)
            by_success[str(succ)]["state"].append(e_state)
            by_success[str(succ)]["dyn"].append(e_dyn)

    errors_state = np.array(errors_state)
    errors_dyn = np.array(errors_dyn)
    mse_state = np.mean(errors_state)
    mse_dyn = np.mean(errors_dyn)
    mae_state = np.mean(np.sqrt(errors_state))
    mae_dyn = np.mean(np.sqrt(errors_dyn))

    print("=== S1 评估 ===")
    print(f"样本数: {len(dataset)}")
    print(f"当前步 target MSE: {mse_state:.6f}  MAE: {mae_state:.6f}")
    print(f"下一步 target MSE: {mse_dyn:.6f}  MAE: {mae_dyn:.6f}")
    print("\n按域:")
    for dn, v in by_domain.items():
        s = np.mean(v["state"])
        d = np.mean(v["dyn"])
        print(f"  {dn}: L_state MSE={s:.6f}  L_dyn MSE={d:.6f}")
    print("\n按 success:")
    for succ, v in by_success.items():
        s = np.mean(v["state"])
        d = np.mean(v["dyn"])
        print(f"  success={succ}: L_state MSE={s:.6f}  L_dyn MSE={d:.6f}")


if __name__ == "__main__":
    main()
