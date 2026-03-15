"""
从 IsaacGym 采集 S1 格式数据。

因 Isaac 依赖 Hydra 与 isaacgymenvs 的 cfg 目录，实际采集由 isaacgymenvs/collect_s1_baoding.py 完成。
本脚本提供从仓库根目录的调用方式（通过 subprocess 切到 isaacgymenvs 并传参）。

用法（在仓库根目录）:
  python -m belief_s1.collect_isaac --out_dir data/s1_baoding --steps 2000
  python -m belief_s1.collect_isaac --checkpoint runs/baoding/nn/baoding.pth --out_dir data/s1_baoding
"""

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ISAAC_SCRIPT = os.path.join(_REPO_ROOT, "isaacgymenvs", "collect_s1_baoding.py")


def parse_args():
    p = argparse.ArgumentParser(description="IsaacGym S1 数据采集（内部调用 isaacgymenvs/collect_s1_baoding.py）")
    p.add_argument("--out_dir", type=str, default="data/s1_baoding", help="输出目录")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--checkpoint", type=str, default=None, help="Teacher .pth 路径")
    p.add_argument("--action_noise", type=float, default=0.05)
    p.add_argument("--num_envs", type=int, default=4, help="Isaac 并行 env 数")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(ISAAC_SCRIPT):
        print(f"未找到 {ISAAC_SCRIPT}，请确认仓库结构。")
        sys.exit(1)
    out_abs = os.path.abspath(args.out_dir)
    overrides = [
        "task=AllegroArmMOAR",
        "task.env.observationType=full_stack_baoding",
        f"task.env.numEnvs={args.num_envs}",
        f"distill.teacher_data_dir={out_abs}",
        f"distill.learn.nsteps={max(1, args.steps // 10)}",
    ]
    if args.checkpoint:
        overrides.append(f"checkpoint={os.path.abspath(args.checkpoint)}")
    cmd = [sys.executable, ISAAC_SCRIPT] + overrides
    cwd = os.path.join(_REPO_ROOT, "isaacgymenvs")
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[belief_s1] 调用: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env)
    print(f"[belief_s1] 输出目录: {out_abs}")


if __name__ == "__main__":
    main()
