"""
S1 导出数据检查脚本：obs_deploy 维度与内容、priv_target 前后步对应、
sin/cos 连续性与归一化、target 数值范围。

用法（仓库根目录）:
  python -m belief_s1.check_data --data_dir data/s1_baoding
  python -m belief_s1.check_data --data_dir data/s1_baoding --npz data/s1_baoding/mujoco_s1_xxx.npz
  python -m belief_s1.check_data --data_dir data/s1_baoding --n_samples 50
"""

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.schema import N_DEPLOY, N_OBS_DIM, N_STACK, N_PRIV_FULL, N_PRIV_TARGET


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def run_checks(data_dir: str, npz_path: str = None, n_samples: int = 20, n_print: int = 5):
    if npz_path and os.path.isfile(npz_path):
        files = [npz_path]
    else:
        import glob
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"[check_data] 未找到 .npz 文件: {data_dir}")
        return

    all_obs, all_pt, all_pt1, all_done, all_meta = [], [], [], [], []
    file_lengths = []
    for f in files:
        d = load_npz(f)
        n = d["obs_deploy"].shape[0]
        file_lengths.append(n)
        all_obs.append(d["obs_deploy"])
        all_pt.append(d["priv_target_t"])
        pt1_key = "priv_target_t_plus_1" if "priv_target_t_plus_1" in d else "priv_target_t1"
        all_pt1.append(d[pt1_key])
        meta = d["meta"]
        done = np.array([meta[i].get("done", False) if isinstance(meta[i], dict) else False for i in range(n)])
        all_done.append(done)
        all_meta.append(meta)
    obs_deploy = np.concatenate(all_obs, axis=0)
    priv_t = np.concatenate(all_pt, axis=0)
    priv_t1 = np.concatenate(all_pt1, axis=0)
    done = np.concatenate(all_done, axis=0)

    n_total = obs_deploy.shape[0]
    # 多文件时，文件边界处 (i, i+1) 不是同一条轨迹，统计时排除
    is_file_boundary = np.zeros(n_total, dtype=bool)
    end = 0
    for L in file_lengths:
        end += L
        if end - 1 < n_total:
            is_file_boundary[end - 1] = True  # 该下标是某文件最后一条，与下一条跨文件
    idx_sample = np.random.RandomState(42).choice(n_total, size=min(n_samples, n_total), replace=False)

    print("=" * 60)
    print("1. obs_deploy 维度和内容")
    print("=" * 60)
    assert obs_deploy.shape[1] == N_DEPLOY, f"期望 obs_deploy 第二维 {N_DEPLOY}，得到 {obs_deploy.shape[1]}"
    print(f"   维度: {obs_deploy.shape} (期望 N={n_total}, 680)")
    if obs_deploy.shape[1] != 680:
        print("   [FAIL] 非 680 维，可能混入 privileged 或栈长不对")
    else:
        print("   [OK] 680 维 = 8 帧 × 85 维，未含 26 维 privileged")
    # 栈顺序：第一段 85 应为“当前帧”，后续为历史
    for ii, i in enumerate(idx_sample[:n_print]):
        frame0 = obs_deploy[i, :85]
        frame1 = obs_deploy[i, 85:170]
        diff = np.abs(frame0 - frame1).max()
        print(f"   样本 {i}: 首帧[0:5]={frame0[:5].round(4)}, 次帧[0:5]={frame1[:5].round(4)}, max|差|={diff:.4f}")
    print("   (首帧=当前帧，次帧=上一帧，应有差异；若 8 帧栈顺序与实现一致则 OK)\n")

    print("=" * 60)
    print("2. priv_target_t 与 priv_target_t_plus_1 前后步对应")
    print("=" * 60)
    # 仅统计同一条轨迹内的连续对：排除 done 与跨文件边界
    gaps = []
    for i in range(n_total - 1):
        if done[i] or is_file_boundary[i]:
            continue
        g = np.abs(priv_t1[i] - priv_t[i + 1])
        gaps.append(g)
    if gaps:
        gaps = np.array(gaps)
        print(f"   非 done 的连续对数目: {len(gaps)}")
        print(f"   |priv_t_plus_1[i] - priv_t[i+1]| 各维: max={gaps.max(axis=0).round(6)}, mean={gaps.mean(axis=0).round(6)}")
        if np.any(gaps.max(axis=0) > 0.1):
            print("   [WARN] 存在较大偏差，请核对采集时是否用 next_obs 正确算 priv_t_plus_1")
        else:
            print("   [OK] 前后步 target 对应良好")
    else:
        print("   [SKIP] 无连续非 done 对（例如全为单步 episode）")
    if len(files) > 1:
        print(f"   (已排除 {is_file_boundary.sum()} 个跨文件边界，仅统计同文件内连续对)")
    # 打印几条：当前 target、下一步 target、下一帧的“当前”target
    print("   抽几条同轨迹连续对:")
    count = 0
    for i in range(n_total - 1):
        if done[i] or is_file_boundary[i] or count >= n_print:
            continue
        count += 1
        print(f"     i={i}  priv_t={priv_t[i].round(4)}  pt1={priv_t1[i].round(4)}  next_priv_t={priv_t[i+1].round(4)}  diff={np.abs(priv_t1[i]-priv_t[i+1]).round(6)}")
    print()

    print("=" * 60)
    print("3. sin_phase, cos_phase 连续性与归一化")
    print("=" * 60)
    sin_p = priv_t[:, 0]
    cos_p = priv_t[:, 1]
    norm2 = sin_p ** 2 + cos_p ** 2
    print(f"   sin^2+cos^2: min={norm2.min():.6f}, max={norm2.max():.6f}, mean={norm2.mean():.6f}")
    if np.all(np.abs(norm2 - 1.0) < 0.01):
        print("   [OK] 近似单位圆")
    else:
        print("   [FAIL] 应接近 1")
    # 相邻（同轨迹）变化，排除 done 与跨文件边界
    dsin = []
    dcos = []
    for i in range(n_total - 1):
        if done[i] or is_file_boundary[i]:
            continue
        dsin.append(abs(sin_p[i + 1] - sin_p[i]))
        dcos.append(abs(cos_p[i + 1] - cos_p[i]))
    if dsin:
        dsin = np.array(dsin)
        dcos = np.array(dcos)
        print(f"   相邻步 |Δsin|: max={dsin.max():.4f}, mean={dsin.mean():.4f}")
        print(f"   相邻步 |Δcos|: max={dcos.max():.4f}, mean={dcos.mean():.4f}")
        if dsin.max() > 1.5 or dcos.max() > 1.5:
            print("   [WARN] 存在明显跳变（可能跨 episode 或角度 -π/π 边界）")
        else:
            print("   [OK] 相邻步变化连续")
    print()

    print("=" * 60)
    print("4. target 数值范围")
    print("=" * 60)
    names = ["sin_phase", "cos_phase", "pair_xy_distance", "z_mean", "z_diff"]
    for j in range(N_PRIV_TARGET):
        v = priv_t[:, j]
        print(f"   {names[j]:20s}: min={v.min():8.4f}, max={v.max():8.4f}, mean={v.mean():8.4f}, std={v.std():8.4f}")
    print("   (pair_xy_distance 约两球水平间距; z_mean 约整体高度; z_diff 两球高差)")
    print()

    print("=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"   总样本数: {n_total}, 文件: {files}")
    print(f"   done=True 比例: {done.mean()*100:.1f}%")


def main():
    p = argparse.ArgumentParser(description="S1 导出数据检查")
    p.add_argument("--data_dir", type=str, default="data/s1_baoding", help="数据目录")
    p.add_argument("--npz", type=str, default=None, help="只检查该 npz 文件（覆盖 data_dir 下的列表）")
    p.add_argument("--n_samples", type=int, default=20, help="随机抽多少条做细查")
    p.add_argument("--n_print", type=int, default=5, help="打印多少条样例")
    args = p.parse_args()
    run_checks(args.data_dir, args.npz, args.n_samples, args.n_print)


if __name__ == "__main__":
    main()
