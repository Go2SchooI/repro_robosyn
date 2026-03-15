"""S1 训练与数据路径等配置。"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .schema import N_DEPLOY, N_PRIV_TARGET


@dataclass
class S1Config:
    # 数据
    data_dir: str = "data/s1_baoding"
    domain: Optional[str] = None  # "isaac" | "mujoco" | None=两者
    max_samples: Optional[int] = None

    # 模型
    obs_dim: int = N_DEPLOY
    action_dim: int = 22
    priv_target_dim: int = N_PRIV_TARGET
    latent_dim: int = 64
    enc_hidden: Tuple[int, ...] = (256, 256)
    dec_hidden: Tuple[int, ...] = (128, 128)
    dyn_hidden: Tuple[int, ...] = (128, 128)

    # 训练
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 100
    device: str = "cuda"
    log_dir: str = "runs/belief_s1"
    save_every: int = 10
