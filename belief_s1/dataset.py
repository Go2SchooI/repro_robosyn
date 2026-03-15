"""
从已导出的 S1 格式数据加载 (obs_deploy, action, priv_t, priv_t+1, meta)。

支持单文件 .npz 或目录下多 .npz/.pt 文件；每条样本为单步 transition。
"""

import os
import glob
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz_samples(path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """加载单个 .npz（内部为多条样本的数组）。"""
    if verbose:
        print("  [dataset] 正在加载: {} ...".format(os.path.basename(path)), flush=True)
    data = np.load(path, allow_pickle=True)
    if "obs_deploy" not in data:
        return []
    n = data["obs_deploy"].shape[0]
    pt1_key = "priv_target_t_plus_1" if "priv_target_t_plus_1" in data else "priv_target_t1"
    has_pd = "pd_targets" in data
    samples = []
    step = max(1, n // 5)  # 最多打印约 5 次进度
    for i in range(n):
        meta = data["meta"][i] if "meta" in data else {}
        if isinstance(meta, np.ndarray):
            meta = meta.item() if meta.ndim == 0 else meta.tolist()
        if not isinstance(meta, dict):
            meta = {}
        s = {
            "obs_deploy": data["obs_deploy"][i],
            "action": data["action"][i],
            "priv_target_t": data["priv_target_t"][i],
            "priv_target_t_plus_1": data[pt1_key][i],
            "meta": meta,
        }
        if has_pd:
            s["pd_targets"] = data["pd_targets"][i]
        samples.append(s)
        if verbose and step > 0 and (i + 1) % step == 0 and (i + 1) < n:
            print("    [dataset] 已读 {}/{} 条".format(i + 1, n), flush=True)
    if verbose:
        print("    [dataset] 完成: {} 条".format(n), flush=True)
    return samples


def load_pt_samples(path: str) -> List[Dict[str, Any]]:
    """加载 .pt（list of dict 或单 dict 含 list）。"""
    x = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        if "samples" in x:
            return x["samples"]
        if "obs_deploy" in x and isinstance(x["obs_deploy"], (list, np.ndarray, torch.Tensor)):
            n = len(x["obs_deploy"]) if hasattr(x["obs_deploy"], "__len__") else x["obs_deploy"].shape[0]
            samples = []
            for i in range(n):
                obs = x["obs_deploy"][i]
                if torch.is_tensor(obs):
                    obs = obs.numpy()
                action = x["action"][i]
                if torch.is_tensor(action):
                    action = action.numpy()
                pt = x["priv_target_t"][i]
                if torch.is_tensor(pt):
                    pt = pt.numpy()
                pt1 = x["priv_target_t_plus_1"][i]
                if torch.is_tensor(pt1):
                    pt1 = pt1.numpy()
                meta = x.get("meta", [{}] * n)
                if isinstance(meta, (list, tuple)) and i < len(meta):
                    m = meta[i]
                else:
                    m = meta if isinstance(meta, dict) else {}
                if hasattr(m, "item"):
                    m = m.item() if hasattr(m, "ndim") and m.ndim == 0 else dict(m)
                samples.append({
                    "obs_deploy": np.asarray(obs, dtype=np.float32),
                    "action": np.asarray(action, dtype=np.float32),
                    "priv_target_t": np.asarray(pt, dtype=np.float32),
                    "priv_target_t_plus_1": np.asarray(pt1, dtype=np.float32),
                    "meta": m if isinstance(m, dict) else {},
                })
            return samples
    return []


def collect_sample_files(data_dir: str, pattern: str = "*.npz") -> List[str]:
    """收集目录下所有 .npz 或 .pt 文件。"""
    files = []
    files.extend(glob.glob(os.path.join(data_dir, "*.npz")))
    if "pt" in pattern or pattern == "*":
        files.extend(glob.glob(os.path.join(data_dir, "*.pt")))
    return sorted(files)


class BeliefS1Dataset(Dataset):
    """S1 用的 PyTorch Dataset：返回 (obs_deploy, action, priv_t, priv_t+1)。"""

    def __init__(
        self,
        data_dir: str,
        domain: Optional[str] = None,
        max_samples: Optional[int] = None,
        load_pt: bool = True,
        load_npz: bool = True,
    ):
        self.samples: List[Dict[str, Any]] = []
        if load_npz:
            for p in collect_sample_files(data_dir, "*.npz"):
                self.samples.extend(load_npz_samples(p, verbose=True))
        if load_pt:
            for p in collect_sample_files(data_dir, "*.pt"):
                self.samples.extend(load_pt_samples(p))
        if domain is not None:
            self.samples = [s for s in self.samples if s.get("meta", {}).get("domain_name") == domain]
        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.Random(42).shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        self.samples = [s for s in self.samples if s["obs_deploy"] is not None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        obs = torch.from_numpy(np.asarray(s["obs_deploy"], dtype=np.float32))
        action = torch.from_numpy(np.asarray(s["action"], dtype=np.float32))
        pt = torch.from_numpy(np.asarray(s["priv_target_t"], dtype=np.float32))
        pt1 = torch.from_numpy(np.asarray(s["priv_target_t_plus_1"], dtype=np.float32))
        done = torch.tensor(float(s.get("meta", {}).get("done", False)), dtype=torch.float32)
        return obs, action, pt, pt1, done
