"""
S1 训练脚本：L_state + L_dyn，仅单域 belief learning。

用法:
  python -m belief_s1.train_s1 --data_dir data/s1_baoding [--domain isaac] [--epochs 100]

超小数据过拟合测试（验证数据/loss/对齐无误）:
  python -m belief_s1.train_s1 --data_dir data/s1_baoding --overfit_test
  # 等价于 max_samples=1024, epochs=500，每 50 epoch 及结束时打印各维 MSE
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 仓库根目录加入 path，便于单独运行
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.config import S1Config
from belief_s1.dataset import BeliefS1Dataset
from belief_s1.models import S1BeliefModel

# 5 维 target 名称，用于 per-dim 统计
PRIV_DIM_NAMES = ("sin_phase", "cos_phase", "pair_xy_distance", "z_mean", "z_diff")


def parse_args():
    p = argparse.ArgumentParser(description="S1 Direct belief learning 训练")
    p.add_argument("--data_dir", type=str, default="data/s1_baoding", help="数据目录（含 .npz/.pt）")
    p.add_argument("--domain", type=str, default=None, choices=["isaac", "mujoco", None], help="仅用某域数据")
    p.add_argument("--max_samples", type=int, default=None, help="最多用多少条样本；过拟合测试建议 512/1024")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overfit_test", action="store_true", help="超小数据过拟合测试：max_samples=1024, epochs=500, 结束时打 per-dim MSE")
    p.add_argument("--no_wandb", action="store_true", help="禁用 wandb 上报")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = "s1_{}_{}".format(args.domain or "all", datetime.now().strftime("%Y%m%d_%H%M%S"))
        wandb.init(project="belief_s1", config=vars(args), name=run_name)
        print("[S1] wandb 已初始化，run: {}".format(wandb.run.name if wandb.run else run_name), flush=True)

    log_per_dim_interval = 0
    if args.overfit_test:
        args.max_samples = args.max_samples or 1024
        if args.epochs == 100:
            args.epochs = 500
        log_per_dim_interval = 50
        print("[S1] 超小数据过拟合测试: max_samples={}, epochs={}".format(args.max_samples, args.epochs), flush=True)

    print("[S1] 正在加载数据...", flush=True)
    dataset = BeliefS1Dataset(
        data_dir=args.data_dir,
        domain=args.domain,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        print(f"[S1] 未找到数据，请先运行 collect 脚本生成数据到 {args.data_dir}", flush=True)
        if use_wandb:
            wandb.finish()
        return
    n_samples = len(dataset)
    print(f"[S1] 加载样本数: {n_samples}", flush=True)
    if use_wandb:
        wandb.log({"info/n_samples": n_samples}, step=0)

    val_ratio = 0.1
    train_indices = list(range(n_samples))
    val_indices = []
    if val_ratio > 0 and n_samples > 10:
        import random
        random.Random(42).shuffle(train_indices)
        n_val = max(1, int(n_samples * val_ratio))
        val_indices = train_indices[-n_val:]
        train_indices = train_indices[:-n_val]
        print("[S1] 训练/验证划分: train {}  val {}".format(len(train_indices), len(val_indices)), flush=True)
    else:
        val_ratio = 0.0

    batch_size = min(args.batch_size, len(train_indices)) if train_indices else args.batch_size
    train_dataset = Subset(dataset, train_indices) if val_indices else dataset
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # 在 log_dir 下按时间建子目录，避免多次训练互相覆盖
    run_dir = os.path.join("runs/belief_s1", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print("[S1] 本次 run 目录: {}".format(run_dir), flush=True)

    cfg = S1Config(
        data_dir=args.data_dir,
        domain=args.domain,
        max_samples=args.max_samples,
        latent_dim=128,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        log_dir=run_dir,
        save_every=100,
        device=device,
    )
    model = S1BeliefModel(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        priv_target_dim=cfg.priv_target_dim,
        latent_dim=128,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    os.makedirs(cfg.log_dir, exist_ok=True)
    print("[S1] 开始训练 (epochs={})...".format(cfg.epochs), flush=True)
    for epoch in range(cfg.epochs):
        model.train()
        total_state_loss = 0.0
        total_dyn_loss = 0.0
        n_batches = 0
        for obs, action, priv_t, priv_t1, done in loader:
            obs = obs.to(device)
            action = action.to(device)
            priv_t = priv_t.to(device)
            priv_t1 = priv_t1.to(device)
            done = done.to(device)
            _, x_hat_t, x_hat_t1 = model(obs, action)
            l_state = mse(x_hat_t, priv_t)
            # 仅对同 episode 内 transition 算 L_dyn：done=True 表示本步后 reset，priv_t_plus_1 非同一轨迹
            dyn_err = (x_hat_t1 - priv_t1).pow(2).sum(dim=-1)
            mask = (1.0 - done)
            n_valid = mask.sum()
            if n_valid > 0:
                l_dyn = (dyn_err * mask).sum() / n_valid
            else:
                l_dyn = torch.tensor(0.0, device=obs.device, dtype=obs.dtype)
            loss = l_state + l_dyn
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_state_loss += l_state.item()
            total_dyn_loss += l_dyn.item()
            n_batches += 1
        avg_s = total_state_loss / max(n_batches, 1)
        avg_d = total_dyn_loss / max(n_batches, 1)
        if use_wandb:
            wandb.log({"train/L_state": avg_s, "train/L_dyn": avg_d}, step=epoch)
        if (epoch + 1) % 20 == 0:
            print(f"[S1] epoch {epoch} L_state={avg_s:.6f} L_dyn={avg_d:.6f}", flush=True)
        if val_ratio > 0 and val_indices and (epoch + 1) % 20 == 0:
            val_s, val_d = _eval_loss(model, dataset, val_indices, device)
            if use_wandb:
                wandb.log({"val/L_state": val_s, "val/L_dyn": val_d}, step=epoch)
            print(f"       val  L_state={val_s:.6f} L_dyn={val_d:.6f}", flush=True)
        if log_per_dim_interval and (epoch + 1) % log_per_dim_interval == 0:
            _log_per_dim_mse(model, dataset, device, args, use_wandb=use_wandb, step=epoch)
        if (epoch + 1) % 100 == 0:
            path = os.path.join(cfg.log_dir, f"s1_ep{epoch+1}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch}, path)
            print(f"  保存: {path}", flush=True)
    path_last = os.path.join(cfg.log_dir, "s1_last.pt")
    torch.save({"model": model.state_dict(), "epoch": cfg.epochs - 1}, path_last)
    print(f"[S1] 训练结束，最后模型: {path_last}", flush=True)

    if args.overfit_test or log_per_dim_interval:
        print("\n[S1] 全量 train set 各维 MSE（当前步 / 下一步）:", flush=True)
        _log_per_dim_mse(model, dataset, device, args, use_wandb=use_wandb, step=cfg.epochs - 1)
    if use_wandb:
        wandb.finish()


def _eval_loss(model, dataset, indices, device):
    """在给定样本下标上算平均 L_state 与 L_dyn（不反传），返回 (avg_L_state, avg_L_dyn)。"""
    model.eval()
    total_s, total_d, n_batch, n_dyn = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for idx in indices:
            obs, action, pt, pt1, done = dataset[idx]
            obs = obs.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            pt = pt.unsqueeze(0).to(device)
            pt1 = pt1.unsqueeze(0).to(device)
            _, x_hat_t, x_hat_t1 = model(obs, action)
            total_s += ((x_hat_t - pt) ** 2).sum().item()
            n_batch += 1
            if done.item() < 0.5:
                total_d += ((x_hat_t1 - pt1) ** 2).sum().item()
                n_dyn += 1
    model.train()
    avg_s = total_s / max(n_batch, 1)
    avg_d = total_d / max(n_dyn, 1)
    return avg_s, avg_d


def _log_per_dim_mse(model, dataset, device, args, use_wandb=False, step=0):
    """在 dataset 上算一遍，打印 L_state / L_dyn 各维 MSE；可选上报 wandb。"""
    model.eval()
    K = len(PRIV_DIM_NAMES)
    sum_state = np.zeros(K)
    sum_dyn = np.zeros(K)
    n_state = 0
    n_dyn = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            obs, action, pt, pt1, done = dataset[i]
            obs = obs.unsqueeze(0).to(device)
            action = action.unsqueeze(0).to(device)
            _, x_hat_t, x_hat_t1 = model(obs, action)
            pt = pt.unsqueeze(0)
            pt1 = pt1.unsqueeze(0)
            sum_state += ((x_hat_t.cpu() - pt) ** 2).numpy().sum(axis=0)
            n_state += 1
            if done.item() < 0.5:
                sum_dyn += ((x_hat_t1.cpu() - pt1) ** 2).numpy().sum(axis=0)
                n_dyn += 1
    mse_state = sum_state / max(n_state, 1)
    mse_dyn = sum_dyn / max(n_dyn, 1)
    print("   L_state 各维 MSE:", dict(zip(PRIV_DIM_NAMES, [round(x, 6) for x in mse_state.tolist()])), flush=True)
    print("   L_dyn   各维 MSE:", dict(zip(PRIV_DIM_NAMES, [round(x, 6) for x in mse_dyn.tolist()])), flush=True)
    print("   L_state 总:", round(float(mse_state.sum()), 6), "  L_dyn 总:", round(float(mse_dyn.sum()), 6), flush=True)
    if use_wandb:
        import wandb
        log_dict = {f"per_dim_state/{name}": val for name, val in zip(PRIV_DIM_NAMES, mse_state.tolist())}
        log_dict.update({f"per_dim_dyn/{name}": val for name, val in zip(PRIV_DIM_NAMES, mse_dyn.tolist())})
        wandb.log(log_dict, step=step)
    model.train()


if __name__ == "__main__":
    main()
