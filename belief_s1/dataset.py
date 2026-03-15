"""
从已导出的 S1 格式数据加载 (obs_deploy, action, priv_t, priv_t+1, meta)。

支持单文件 .npz 或目录下多 .npz/.pt 文件；每条样本为单步 transition。
- 按块存储（整块数组），不建 N 个 dict，避免内存暴增和卡死。
- max_samples 在加载时即截断：只从首个（及后续）文件中读取所需条数，不先全量再截断。
- 注意：单个 .npz 在 np.load() 时仍会整文件读入内存一次；若首文件极大，建议拆成多个小 npz 或单独放一个小文件用于过拟合测试。
"""

import os
import glob
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _extract_done(meta_arr: Any, n: int) -> np.ndarray:
    """从 meta 数组提取每行的 done 标志，返回 (n,) bool。"""
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        m = meta_arr[i] if hasattr(meta_arr, "__getitem__") else {}
        if isinstance(m, np.ndarray):
            m = m.item() if m.ndim == 0 else (m.tolist() if m.size else {})
        if not isinstance(m, dict):
            continue
        out[i] = bool(m.get("done", False))
    return out


def load_npz_chunk(
    path: str,
    max_rows: Optional[int] = None,
    domain: Optional[str] = None,
    verbose: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    加载单个 .npz 为一块数组（不建 N 个 dict）。
    若 max_rows 有值则只取前 max_rows 行；若 domain 有值则只保留 meta['domain_name']==domain 的行。
    返回 None 表示无有效数据；否则返回 dict: obs_deploy, action, priv_target_t, priv_target_t_plus_1, done (均为 numpy 数组)。
    """
    if verbose:
        print("  [dataset] 正在加载: {} ...".format(os.path.basename(path)), flush=True)
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        if verbose:
            print("  [dataset] 读取失败: {}".format(e), flush=True)
        return None
    if "obs_deploy" not in data:
        return None
    n = data["obs_deploy"].shape[0]
    pt1_key = "priv_target_t_plus_1" if "priv_target_t_plus_1" in data else "priv_target_t1"
    # 先取全部或前 max_rows，再按 domain 过滤；拷贝切片后立即释放 data，避免长期占用大文件内存
    take = min(n, max_rows) if max_rows is not None else n
    obs = np.asarray(data["obs_deploy"][:take], dtype=np.float32).copy()
    action = np.asarray(data["action"][:take], dtype=np.float32).copy()
    priv_t = np.asarray(data["priv_target_t"][:take], dtype=np.float32).copy()
    priv_t1 = np.asarray(data[pt1_key][:take], dtype=np.float32).copy()
    meta_arr = data["meta"][:take].copy() if "meta" in data else None
    del data
    if domain is not None and meta_arr is not None:
        mask = np.zeros(take, dtype=np.bool_)
        for i in range(take):
            m = meta_arr[i]
            if isinstance(m, np.ndarray):
                m = m.item() if m.ndim == 0 else (m.tolist() if m.size else {})
            if isinstance(m, dict) and m.get("domain_name") == domain:
                mask[i] = True
        if not np.any(mask):
            return None
        obs = obs[mask]
        action = action[mask]
        priv_t = priv_t[mask]
        priv_t1 = priv_t1[mask]
        meta_arr = meta_arr[mask]
        take = obs.shape[0]
    done = _extract_done(meta_arr, take) if meta_arr is not None else np.zeros(take, dtype=np.bool_)
    if verbose:
        print("    [dataset] 完成: {} 条".format(take), flush=True)
    return {
        "obs_deploy": obs,
        "action": action,
        "priv_target_t": priv_t,
        "priv_target_t_plus_1": priv_t1,
        "done": done,
    }


def load_npz_samples(path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """兼容旧接口：加载单个 .npz 为 list of dict（仅在不使用 max_samples 且需 list 时用，避免大文件）。"""
    chunk = load_npz_chunk(path, max_rows=None, domain=None, verbose=verbose)
    if chunk is None:
        return []
    n = chunk["obs_deploy"].shape[0]
    samples = []
    for i in range(n):
        samples.append({
            "obs_deploy": chunk["obs_deploy"][i],
            "action": chunk["action"][i],
            "priv_target_t": chunk["priv_target_t"][i],
            "priv_target_t_plus_1": chunk["priv_target_t_plus_1"][i],
            "meta": {"done": bool(chunk["done"][i])},
        })
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
    """
    S1 用的 PyTorch Dataset：返回 (obs_deploy, action, priv_t, priv_t+1, done)。
    内部按「块」存数组，不建 N 个 dict；且当 max_samples 有值时在加载阶段即只读所需数量，避免全量加载大目录。
    """

    def __init__(
        self,
        data_dir: str,
        domain: Optional[str] = None,
        max_samples: Optional[int] = None,
        load_pt: bool = True,
        load_npz: bool = True,
    ):
        self.chunks: List[Dict[str, np.ndarray]] = []
        self.cumlen: List[int] = [0]
        npz_paths = collect_sample_files(data_dir, "*.npz") if load_npz else []
        total = 0
        for p in npz_paths:
            if max_samples is not None and total >= max_samples:
                break
            remaining = (max_samples - total) if max_samples is not None else None
            chunk = load_npz_chunk(p, max_rows=remaining, domain=domain, verbose=True)
            if chunk is None or chunk["obs_deploy"].shape[0] == 0:
                continue
            n = chunk["obs_deploy"].shape[0]
            self.chunks.append(chunk)
            total += n
            self.cumlen.append(total)
        if load_pt:
            pt_paths = collect_sample_files(data_dir, "*.pt")
            for p in pt_paths:
                if max_samples is not None and total >= max_samples:
                    break
                samples = load_pt_samples(p)
                if not samples:
                    continue
                if domain is not None:
                    samples = [s for s in samples if s.get("meta", {}).get("domain_name") == domain]
                if max_samples is not None:
                    samples = samples[: max_samples - total]
                if not samples:
                    continue
                n = len(samples)
                chunk = {
                    "obs_deploy": np.array([s["obs_deploy"] for s in samples], dtype=np.float32),
                    "action": np.array([s["action"] for s in samples], dtype=np.float32),
                    "priv_target_t": np.array([s["priv_target_t"] for s in samples], dtype=np.float32),
                    "priv_target_t_plus_1": np.array([s["priv_target_t_plus_1"] for s in samples], dtype=np.float32),
                    "done": np.array([s.get("meta", {}).get("done", False) for s in samples], dtype=np.bool_),
                }
                self.chunks.append(chunk)
                total += n
                self.cumlen.append(total)

    def __len__(self) -> int:
        return self.cumlen[-1] if self.cumlen else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 找到所在块与块内下标（cumlen[0]=0, cumlen[1]=n1, ...）
        cidx = 0
        for i in range(len(self.cumlen) - 1):
            if idx < self.cumlen[i + 1]:
                cidx = i
                break
        else:
            cidx = len(self.cumlen) - 2
        local = idx - self.cumlen[cidx]
        c = self.chunks[cidx]
        obs = torch.from_numpy(np.asarray(c["obs_deploy"][local], dtype=np.float32))
        action = torch.from_numpy(np.asarray(c["action"][local], dtype=np.float32))
        pt = torch.from_numpy(np.asarray(c["priv_target_t"][local], dtype=np.float32))
        pt1 = torch.from_numpy(np.asarray(c["priv_target_t_plus_1"][local], dtype=np.float32))
        done = torch.tensor(float(c["done"][local]), dtype=torch.float32)
        return obs, action, pt, pt1, done
