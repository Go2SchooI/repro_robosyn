"""
统一 Sim2Sim 数据 schema：deployable obs、action、privileged targets、meta。

与 isaacgymenvs/tasks (full_stack_baoding) 和 mujoco_sim/observations 对齐：
- 单帧 85 维，栈 8 帧 → deployable 680 维，privileged 26 维（两球 pose+vel）。
- 第一版 privileged target：5 维，基于两球相对几何与高度，不依赖轨道中心。
"""

import numpy as np
from typing import Dict, Any, Optional

# ----- 与现有 env 一致的常量 -----
N_OBS_DIM = 85
N_STACK = 8
N_DEPLOY = N_OBS_DIM * N_STACK  # 680
N_PRIV_FULL = 26   # 两球各 13：pos(3)+quat(4)+linvel(3)+angvel*0.2(3)
N_BALLS = 2

# 第一版：5 维 target（低维、连续、与任务进度/稳定性直接相关、不依赖局部原点）
# [sin_phase, cos_phase, pair_xy_distance, z_mean, z_diff]
# phi = atan2(y2-y1, x2-x1)，sin_phase/cos_phase 避免角度周期跳变
N_PRIV_TARGET = 5

# Meta 字段名（success 缺省 -1；spin_progress 作为标量写入 meta 供分层/retrieval 用）
META_KEYS = ("domain_name", "episode_id", "timestep", "success", "task_phase", "spin_progress")


def deploy_slice_from_full_obs(obs: np.ndarray) -> np.ndarray:
    """从 706 维完整 obs 中取出 deployable 部分（前 680 维）。"""
    assert obs.shape[-1] == N_DEPLOY + N_PRIV_FULL, (
        f"obs 应为 706 维，得到 {obs.shape[-1]}")
    return np.asarray(obs[..., :N_DEPLOY], dtype=np.float32)


def priv_full_slice_from_full_obs(obs: np.ndarray) -> np.ndarray:
    """从 706 维完整 obs 中取出完整 privileged 26 维。"""
    assert obs.shape[-1] == N_DEPLOY + N_PRIV_FULL
    return np.asarray(obs[..., N_DEPLOY:], dtype=np.float32)


def compute_priv_target_from_full_priv(priv_26: np.ndarray) -> np.ndarray:
    """
    从 26 维 privileged（两球 pose+vel）算出 5 维 target。

    定义（双球绕世界系固定 z 轴旋转任务）：
        p1=(x1,y1,z1), p2=(x2,y2,z2) 为两球球心世界坐标
        phi = atan2(y2-y1, x2-x1)  两球连线在 xy 平面方向角
        sin_phase = sin(phi), cos_phase = cos(phi)  无跳变相位表示
        pair_xy_distance = sqrt((x2-x1)^2 + (y2-y1)^2)
        z_mean = (z1+z2)/2,  z_diff = z1-z2

    priv_26 布局（与 Isaac/MuJoCo 一致）:
        [0:3] ball1 pos, [3:7] ball1 quat, [7:10] ball2 pos, [10:14] ball2 quat,
        [14:17] ball1 linvel, [17:20] ball2 linvel, [20:26] angvel*0.2
    """
    x1, y1, z1 = priv_26[0], priv_26[1], priv_26[2]
    x2, y2, z2 = priv_26[7], priv_26[8], priv_26[9]
    dx = x2 - x1
    dy = y2 - y1
    phi = np.arctan2(dy, dx)
    sin_phase = np.sin(phi)
    cos_phase = np.cos(phi)
    pair_xy_distance = np.sqrt(dx * dx + dy * dy)
    z_mean = (z1 + z2) * 0.5
    z_diff = z1 - z2
    out = np.array([sin_phase, cos_phase, pair_xy_distance, z_mean, z_diff], dtype=np.float32)
    return out


def compute_priv_target_from_deploy_frame(frame_85: np.ndarray) -> float:
    """
    从单帧 85 维中取 FSR 段 [45:61] 求和，作为 contact_sum。
    用于有 deploy 但无单独 contact 信号时。
    """
    if frame_85.shape[-1] < 61:
        return 0.0
    return float(np.sum(frame_85[..., 45:61]))


def sample_to_dict(
    obs_deploy: np.ndarray,
    action: np.ndarray,
    priv_target_t: np.ndarray,
    priv_target_t_plus_1: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """单条 transition 的规范 dict，便于序列化与 Dataset 加载。"""
    return {
        "obs_deploy": np.asarray(obs_deploy, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "priv_target_t": np.asarray(priv_target_t, dtype=np.float32),
        "priv_target_t_plus_1": np.asarray(priv_target_t_plus_1, dtype=np.float32),
        "meta": dict(meta),
    }


def get_meta_template(
    domain_name: str,
    episode_id: int,
    timestep: int,
    success: int = -1,
    spin_progress: Optional[float] = None,
) -> Dict[str, Any]:
    """返回一条样本的 meta 模板。success 缺省 -1；spin_progress 由 env 暴露时写入供分层/retrieval。"""
    meta = {
        "domain_name": domain_name,
        "episode_id": episode_id,
        "timestep": timestep,
        "success": success,
        "task_phase": -1.0,
    }
    if spin_progress is not None:
        meta["spin_progress"] = float(spin_progress)
    return meta
