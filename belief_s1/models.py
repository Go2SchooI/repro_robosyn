"""
S1: Direct belief learning 模型。

E: obs_deploy -> z
D: z -> priv_target_t
G: (z, action) -> priv_target_t_plus_1

预留：z 可扩展为 (z_shared, z_private)，D/G 仅用 z_shared 或 concat。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .schema import N_DEPLOY, N_PRIV_TARGET


class Encoder(nn.Module):
    """E: deployable observation history -> latent z."""

    def __init__(
        self,
        input_dim: int = N_DEPLOY,
        latent_dim: int = 64,
        hidden: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, latent_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs_deploy: torch.Tensor) -> torch.Tensor:
        # obs_deploy: (B, 680) 或 (B, 8, 85)
        if obs_deploy.dim() == 3:
            obs_deploy = obs_deploy.flatten(1)
        return self.mlp(obs_deploy)


class Decoder(nn.Module):
    """D: z -> priv_target (K 维)."""

    def __init__(
        self,
        latent_dim: int = 64,
        out_dim: int = N_PRIV_TARGET,
        hidden: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers = []
        d = latent_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class DynamicsHead(nn.Module):
    """G: (z, action) -> priv_target_t_plus_1."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 22,
        out_dim: int = N_PRIV_TARGET,
        hidden: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        inp = latent_dim + action_dim
        layers = []
        d = inp
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.mlp(x)


class S1BeliefModel(nn.Module):
    """E + D + G 组合，用于 S1 训练与评估。"""

    def __init__(
        self,
        obs_dim: int = N_DEPLOY,
        action_dim: int = 22,
        priv_target_dim: int = N_PRIV_TARGET,
        latent_dim: int = 64,
        enc_hidden: Tuple[int, ...] = (256, 256),
        dec_hidden: Tuple[int, ...] = (128, 128),
        dyn_hidden: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, enc_hidden)
        self.decoder = Decoder(latent_dim, priv_target_dim, dec_hidden)
        self.dynamics = DynamicsHead(latent_dim, action_dim, priv_target_dim, dyn_hidden)
        self.latent_dim = latent_dim
        self.priv_target_dim = priv_target_dim

    def forward(
        self,
        obs_deploy: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z = self.encoder(obs_deploy)
        x_hat_t = self.decoder(z)
        x_hat_t_plus_1 = self.dynamics(z, action) if action is not None else None
        return z, x_hat_t, x_hat_t_plus_1
