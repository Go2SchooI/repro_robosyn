"""M4: Lightweight policy network for checkpoint loading and inference.

Reconstructs the rl_games A2C network (MLP [512,256,256] + ELU) without
depending on rl_games, and loads weights from a .pth checkpoint.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class RunningMeanStd(nn.Module):
    """Observation normalizer matching rl_games RunningMeanStd."""

    def __init__(self, insize: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.float()
        var = self.running_var.float()
        y = (x - mean) / torch.sqrt(var + self.epsilon)
        return torch.clamp(y, -5.0, 5.0)


class PolicyNetwork(nn.Module):
    """Minimal actor network matching the rl_games A2C continuous policy.

    Architecture (from AllegroArmMOARPPO.yaml):
        actor_mlp: Linear(obs_dim, 512) -> ELU -> Linear(512, 256) -> ELU
                   -> Linear(256, 256) -> ELU
        mu:        Linear(256, act_dim)  (no activation)
    """

    def __init__(self, obs_dim: int = 366, act_dim: int = 22):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.running_mean_std = RunningMeanStd(obs_dim)

        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.mu = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward: normalize -> MLP -> mu -> clamp."""
        x = self.running_mean_std(obs)
        x = self.actor_mlp(x)
        action = self.mu(x)
        return torch.clamp(action, -1.0, 1.0)

    @staticmethod
    def from_checkpoint(path: str, obs_dim: int = 366, act_dim: int = 22,
                        device: str = "cpu") -> "PolicyNetwork":
        """Load a trained policy from an rl_games .pth checkpoint.

        The checkpoint contains a state_dict with keys like:
            a2c_network.actor_mlp.{0,2,4}.{weight,bias}
            a2c_network.mu.{weight,bias}
            running_mean_std.running_{mean,var}
            running_mean_std.count
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        ckpt_sd = checkpoint["model"]

        policy = PolicyNetwork(obs_dim, act_dim).to(device)

        new_sd = {}
        prefix_map = {
            "a2c_network.actor_mlp.": "actor_mlp.",
            "a2c_network.mu.": "mu.",
            "running_mean_std.": "running_mean_std.",
        }
        for key, value in ckpt_sd.items():
            for ckpt_prefix, local_prefix in prefix_map.items():
                if key.startswith(ckpt_prefix):
                    local_key = local_prefix + key[len(ckpt_prefix):]
                    new_sd[local_key] = value
                    break

        missing, unexpected = policy.load_state_dict(new_sd, strict=False)

        skipped = [k for k in missing if "sigma" not in k and "value" not in k]
        if skipped:
            print(f"[PolicyNetwork] Warning: missing keys: {skipped}")
        if unexpected:
            print(f"[PolicyNetwork] Warning: unexpected keys: {unexpected}")

        policy.eval()
        print(f"[PolicyNetwork] Loaded from {path}")
        print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
        print(f"  running_mean range: [{policy.running_mean_std.running_mean.min():.4f}, "
              f"{policy.running_mean_std.running_mean.max():.4f}]")
        return policy


def test_policy(checkpoint_path: Optional[str] = None):
    """Quick sanity check: random input -> output range."""
    if checkpoint_path:
        policy = PolicyNetwork.from_checkpoint(checkpoint_path)
    else:
        policy = PolicyNetwork()
        print("[test] Using randomly initialized network (no checkpoint)")

    obs = torch.randn(1, 366)
    with torch.no_grad():
        action = policy(obs)

    print(f"  Input shape:  {obs.shape}")
    print(f"  Output shape: {action.shape}")
    print(f"  Output range: [{action.min().item():.4f}, {action.max().item():.4f}]")
    assert action.shape == (1, 22), f"Expected (1,22), got {action.shape}"
    assert action.abs().max() <= 1.0, "Actions should be in [-1, 1]"
    print("  PASS")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    test_policy(ckpt)
