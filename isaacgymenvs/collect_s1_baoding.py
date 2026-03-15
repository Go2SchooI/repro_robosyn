"""
从 IsaacGym 采集 S1 格式数据（obs_deploy, action, priv_t, priv_t+1, meta）。

需在 isaacgymenvs 目录下运行，或从仓库根目录：
  python isaacgymenvs/collect_s1_baoding.py task=AllegroArmMOAR task.env.observationType=full_stack_baoding ...

或通过 belief_s1 入口（见 belief_s1/collect_isaac.py 的说明）。
"""

import os
import sys

import numpy as np
import torch

# 仓库根目录，便于导入 belief_s1
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from belief_s1.schema import (
    N_DEPLOY,
    N_PRIV_FULL,
    deploy_slice_from_full_obs,
    compute_priv_target_from_full_priv,
    get_meta_template,
)


def run_collect(cfg, out_dir: str, num_steps: int, checkpoint_path: str = None, action_noise: float = 0.0):
    """在已有 Hydra cfg 下创建 env、可选加载 policy、rollout 并保存 S1 数据。"""
    from rl_games.common import env_configurations, vecenv
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, get_rlgames_env_creator
    from omegaconf import OmegaConf
    from utils.reformat import omegaconf_to_dict

    cfg_dict = omegaconf_to_dict(cfg.task)
    headless = getattr(cfg, "headless", True)

    def create_env_thunk(**kwargs):
        import isaacgymenvs
        return isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            getattr(cfg, "graphics_device_id", -1),
            headless,
            getattr(cfg, "multi_gpu", False),
            False,
            getattr(cfg, "force_render", False),
            cfg,
            **kwargs,
        )

    vecenv.register("RLGPU", lambda config_name, num_actors, **kw: RLGPUEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": create_env_thunk})

    num_actors = int(OmegaConf.select(cfg, "task.env.numEnvs", default=4))
    vec_env = vecenv.create_vec_env("rlgpu", num_actors)

    policy = None
    device = cfg.rl_device
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            from rl_games.algos_torch import model_builder
            from rl_games.algos_torch import torch_ext
            env_info = vec_env.get_env_info()
            obs_space = env_info.get("observation_space")
            if hasattr(obs_space, "shape"):
                obs_shape = obs_space.shape
            else:
                obs_shape = env_info.get("obs_shape", (706,))
            if isinstance(obs_shape, dict):
                obs_shape = obs_shape.get("obs", (706,))
            train_cfg = omegaconf_to_dict(cfg.train)
            teacher_params = {"config": train_cfg}
            builder = model_builder.ModelBuilder()
            network = builder.load(teacher_params)
            build_config = {
                "actions_num": env_info.get("actions_num", 22),
                "input_shape": obs_shape,
                "num_seqs": num_actors,
                "value_size": env_info.get("value_size", 1),
                "normalize_value": False,
                "normalize_input": train_cfg.get("normalize_input", True),
            }
            actor_critic = network.build(build_config)
            torch_ext.load_checkpoint(checkpoint_path, actor_critic)
            actor_critic.to(device)
            actor_critic.eval()
            policy = actor_critic
            print(f"[Isaac S1] 已加载策略: {checkpoint_path}")
        except Exception as e:
            print(f"[Isaac S1] 加载策略失败，使用随机动作: {e}")
            policy = None

    os.makedirs(out_dir, exist_ok=True)
    samples = []
    current_obs = vec_env.reset()
    episode_id = 0
    step_in_ep = 0
    num_envs = vec_env.env.num_envs

    for step in range(num_steps):
        # current_obs: dict with "obs" -> dict with "obs" (706,) per env
        full_obs = current_obs["obs"]["obs"]
        if torch.is_tensor(full_obs):
            full_obs = full_obs.cpu().numpy()
        # 只取 env 0 的 transition，避免重复
        obs_np = full_obs[0]
        obs_deploy = deploy_slice_from_full_obs(obs_np)
        priv_full = obs_np[N_DEPLOY : N_DEPLOY + N_PRIV_FULL]
        priv_t = compute_priv_target_from_full_priv(priv_full)

        if policy is not None:
            obs_t = torch.from_numpy(full_obs).float().to(device)
            with torch.no_grad():
                res = policy({"obs": obs_t, "is_train": False, "rnn_states": None})
                action = res["actions"][0].cpu().numpy()
        else:
            action = np.zeros(22, dtype=np.float32)
        if action_noise > 0:
            action = action + np.random.randn(22).astype(np.float32) * action_noise
        action = np.clip(action, -1.0, 1.0)
        action_batch = torch.from_numpy(action).float().unsqueeze(0).to(device)
        action_batch = action_batch.repeat(num_envs, 1)

        next_obs, rews, dones, infos = vec_env.step(action_batch)
        next_full = next_obs["obs"]["obs"]
        if torch.is_tensor(next_full):
            next_full = next_full.cpu().numpy()
        next_np = next_full[0]
        next_priv = next_np[N_DEPLOY : N_DEPLOY + N_PRIV_FULL]
        priv_t_plus_1 = compute_priv_target_from_full_priv(next_priv)

        spin_progress = None
        if isinstance(infos, dict) and "spin_progress" in infos:
            sp = infos["spin_progress"]
            try:
                spin_progress = float(sp[0]) if hasattr(sp, "__getitem__") else float(sp)
            except Exception:
                pass
        meta = get_meta_template("isaac", episode_id, step_in_ep, success=-1, spin_progress=spin_progress)
        samples.append({
            "obs_deploy": obs_deploy.copy(),
            "action": action.copy(),
            "priv_target_t": priv_t.copy(),
            "priv_target_t_plus_1": priv_t_plus_1.copy(),
            "meta": meta,
        })

        current_obs = next_obs
        step_in_ep += 1
        if dones.any():
            for i in range(num_envs):
                if dones[i]:
                    episode_id += 1
            step_in_ep = 0 if dones[0] else step_in_ep

        if (step + 1) % 500 == 0:
            print(f"[Isaac S1] step {step+1}/{num_steps}  samples={len(samples)}")

    obs_deploy = np.array([s["obs_deploy"] for s in samples], dtype=np.float32)
    action = np.array([s["action"] for s in samples], dtype=np.float32)
    priv_t = np.array([s["priv_target_t"] for s in samples], dtype=np.float32)
    priv_t1 = np.array([s["priv_target_t_plus_1"] for s in samples], dtype=np.float32)
    meta = np.array([s["meta"] for s in samples], dtype=object)
    out_path = os.path.join(out_dir, "isaac_s1.npz")
    np.savez_compressed(out_path, obs_deploy=obs_deploy, action=action, priv_target_t=priv_t, priv_target_t_plus_1=priv_t1, meta=meta)
    print(f"[Isaac S1] 已保存 {len(samples)} 条 -> {out_path}")
    return out_path


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(config_name="config_distill", config_path="./cfg")
    def main_hydra(cfg: DictConfig):
        out_dir = os.path.abspath(OmegaConf.select(cfg, "distill.teacher_data_dir", default="data/s1_baoding"))
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.getcwd(), out_dir)
        num_steps = int(OmegaConf.select(cfg, "distill.learn.nsteps", default=200)) * 10
        ckpt = OmegaConf.select(cfg, "checkpoint", default=None) or None
        if ckpt:
            ckpt = os.path.abspath(ckpt)
        run_collect(cfg, out_dir=out_dir, num_steps=num_steps, checkpoint_path=ckpt, action_noise=0.05)

    main_hydra()
