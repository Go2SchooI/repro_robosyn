"""M2: MuJoCo environment wrapper for the baoding ball sim2sim task.

Implements:
  - PD torque control (p=100, d=4, clip ±20Nm)
  - Relative control with action smoothing (scale=0.2, moving_average=0.8)
  - 6 PD updates x 2 substeps per control step (match Isaac: control_freq_inv=6, substeps=2)
  - FSR contact sensing
"""

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

from mujoco_sim.utils import (
    HAND_JOINT_NAMES, ARM_JOINT_NAMES, FSR_GEOM_NAMES,
    quat_wxyz_to_xyzw, unscale, scale,
)
from mujoco_sim.observations import ObservationBuilder


# Arm default positions (from the tilted URDF locked joints)
DEFAULT_ARM_QPOS = np.array([0.00, 0.673, -0.916, 3.1416, 2.263, -1.569])

# Hand default positions for baoding (object_set_id == "ball")
DEFAULT_HAND_QPOS_OVERRIDE = {
    "joint_0.0":  0.0,
    "joint_1.0":  1.0,
    "joint_2.0":  0.0,
    "joint_3.0":  0.0,
    "joint_4.0":  0.0048,
    "joint_5.0":  1.0,
    "joint_6.0":  0.0,
    "joint_7.0":  0.0,
    "joint_8.0":  0.0,
    "joint_9.0":  1.0,
    "joint_10.0": 0.0,
    "joint_11.0": 0.0,
    "joint_12.0": 1.3815,
    "joint_13.0": 0.0868,
    "joint_14.0": 0.1259,
    "joint_15.0": 0.0,
}

# Hand joint limits from the URDF (in order of HAND_JOINT_NAMES)
HAND_LOWER_LIMITS = np.array([
    -0.47, -0.196, -0.174, -0.227,   # finger 0
    -0.47, -0.196, -0.174, -0.227,   # finger 1
    -0.47, -0.196, -0.174, -0.227,   # finger 2
     0.70,  0.30,  -0.189, -0.162,   # thumb
])
HAND_UPPER_LIMITS = np.array([
    0.47, 1.61, 1.709, 1.618,   # finger 0
    0.47, 1.61, 1.709, 1.618,   # finger 1
    0.47, 1.61, 1.709, 1.618,   # finger 2
    1.396, 1.163, 1.644, 1.719, # thumb
])

# Full 22-DOF limits (arm + hand)
ARM_LOWER_LIMITS = np.array([0.00, 0.673, -0.91601, 3.1416, 2.263, -1.56901])
ARM_UPPER_LIMITS = np.array([0.00001, 0.6730001, -0.916, 3.14161, 2.2631, -1.569])

ALL_LOWER = np.concatenate([ARM_LOWER_LIMITS, HAND_LOWER_LIMITS])
ALL_UPPER = np.concatenate([ARM_UPPER_LIMITS, HAND_UPPER_LIMITS])



class BaodingMujocoEnv:
    """Single-environment MuJoCo wrapper for sim2sim inference."""

    def __init__(
        self,
        xml_path: str,
        p_gain: float = 100.0,
        d_gain: float = 4.0,
        torque_clip: float = 20.0,
        relative_scale: float = 0.2,
        act_moving_average: float = 0.8,
        control_freq_inv: int = 6,
        substeps: int = 2,
        render: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.p_gain = p_gain
        self.d_gain = d_gain
        self.torque_clip = torque_clip
        self.relative_scale = relative_scale
        self.act_moving_average = act_moving_average
        self.control_freq_inv = control_freq_inv
        self.substeps = substeps
        self.physics_dt = self.model.opt.timestep
        # Isaac: self.dt = sim_params.dt = substeps * physics_dt, used for finite-diff velocity
        self.dt = self.substeps * self.physics_dt

        self._resolve_indices()

        self.obs_builder = ObservationBuilder(
            hand_lower_limits=HAND_LOWER_LIMITS,
            hand_upper_limits=HAND_UPPER_LIMITS,
            arm_lower_limits=ARM_LOWER_LIMITS,
            arm_upper_limits=ARM_UPPER_LIMITS,
        )

        self.prev_targets = np.zeros(22)
        self.last_actions = np.zeros(22)

        self.viewer = None
        if render:
            self._init_viewer()

    def _resolve_indices(self):
        """Resolve MuJoCo model IDs for joints, actuators, geoms."""
        self.arm_jnt_ids = []
        self.arm_qpos_idx = []
        self.arm_qvel_idx = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found"
            self.arm_jnt_ids.append(jid)
            self.arm_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.arm_qvel_idx.append(self.model.jnt_dofadr[jid])

        self.hand_jnt_ids = []
        self.hand_qpos_idx = []
        self.hand_qvel_idx = []
        self.hand_act_ids = []
        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found"
            self.hand_jnt_ids.append(jid)
            self.hand_qpos_idx.append(self.model.jnt_qposadr[jid])
            self.hand_qvel_idx.append(self.model.jnt_dofadr[jid])

            act_name = f"act_{name}"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            assert aid >= 0, f"Actuator '{act_name}' not found"
            self.hand_act_ids.append(aid)

        self.arm_qpos_idx = np.array(self.arm_qpos_idx)
        self.arm_qvel_idx = np.array(self.arm_qvel_idx)
        self.hand_qpos_idx = np.array(self.hand_qpos_idx)
        self.hand_qvel_idx = np.array(self.hand_qvel_idx)
        self.hand_act_ids = np.array(self.hand_act_ids)

        self.all_qpos_idx = np.concatenate([self.arm_qpos_idx, self.hand_qpos_idx])
        self.all_qvel_idx = np.concatenate([self.arm_qvel_idx, self.hand_qvel_idx])

        self.ball_body_ids = []
        self.ball_jnt_ids = []
        self.ball_qpos_idx = []
        self.ball_qvel_idx = []
        for bname in ["ball1", "ball2"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            assert bid >= 0, f"Body '{bname}' not found"
            self.ball_body_ids.append(bid)

            jname = f"{bname}_joint"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"Joint '{jname}' not found"
            self.ball_jnt_ids.append(jid)
            qa = self.model.jnt_qposadr[jid]
            va = self.model.jnt_dofadr[jid]
            self.ball_qpos_idx.append(np.arange(qa, qa + 7))  # pos(3) + quat(4)
            self.ball_qvel_idx.append(np.arange(va, va + 6))  # linvel(3) + angvel(3)

        self.fsr_geom_ids = []
        for gname in FSR_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self.fsr_geom_ids.append(gid)
            else:
                print(f"  Warning: FSR geom '{gname}' not found, will use zero contact")
                self.fsr_geom_ids.append(-1)

    def _init_viewer(self):
        try:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
        except Exception as e:
            print(f"Could not launch viewer: {e}")
            self.viewer = None

    def reset(self) -> np.ndarray:
        """Reset to default pose (from keyframe) and return initial observation."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        default_hand_qpos = np.array([DEFAULT_HAND_QPOS_OVERRIDE[n] for n in HAND_JOINT_NAMES])
        self.prev_targets[:6] = DEFAULT_ARM_QPOS
        self.prev_targets[6:] = default_hand_qpos
        self.last_actions[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.obs_builder.reset()
        obs = self._get_obs()
        return obs

    def _run_pd_loop(self, targets: np.ndarray):
        """Inner PD control loop matching Isaac Gym's step structure.

        Isaac does: for i in range(control_freq_inv=6):
                        update_controller()     # PD with vel = (pos - prev) / dt
                        gym.simulate()          # substeps=2 physics steps

        Here we replicate: 6 PD updates, each followed by 2 mj_step calls.
        self.dt = substeps * physics_dt = 0.01667 (used for finite-diff velocity, same as Isaac).
        """
        prev_qpos = self.data.qpos[self.all_qpos_idx].copy()
        for _ in range(self.control_freq_inv):
            cur_qpos = self.data.qpos[self.all_qpos_idx].copy()
            dof_vel = (cur_qpos - prev_qpos) / self.dt

            torques = self.p_gain * (targets - cur_qpos) - self.d_gain * dof_vel
            torques = np.clip(torques, -self.torque_clip, self.torque_clip)

            self.data.ctrl[self.hand_act_ids] = torques[6:]
            prev_qpos = cur_qpos

            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Execute one control step (control_freq_inv PD updates × substeps physics steps).

        Args:
            actions: (22,) array in [-1, 1], from the policy network.

        Returns:
            obs: (366,) observation vector.
        """
        smoothed_actions = (
            actions * self.act_moving_average
            + self.last_actions * (1.0 - self.act_moving_average)
        )
        targets = self.prev_targets + self.relative_scale * smoothed_actions
        targets = np.clip(targets, ALL_LOWER, ALL_UPPER)

        self.prev_targets = targets.copy()
        self.last_actions = smoothed_actions.copy()

        self._run_pd_loop(targets)

        obs = self._get_obs()
        return obs

    def step_from_targets(self, targets: np.ndarray) -> np.ndarray:
        """Execute one control step using given joint targets (offline trajectory tracking).

        Args:
            targets: (22,) array of desired positions (6 arm + 16 hand), in rad.
                     Same order as CSV from Isaac Gym recordEnv0TrajectoryCsv.

        Returns:
            obs: (366,) observation vector.
        """
        targets = np.asarray(targets, dtype=np.float64)
        assert targets.shape == (22,), f"targets must be (22,), got {targets.shape}"
        targets = np.clip(targets, ALL_LOWER, ALL_UPPER)
        self.prev_targets = targets.copy()

        self._run_pd_loop(targets)

        obs = self._get_obs()
        return obs

    def _get_obs(self) -> np.ndarray:
        """Build the 366-dim observation from current MuJoCo state."""
        hand_qpos = self.data.qpos[self.hand_qpos_idx]
        fsr_contacts = self._get_fsr_contacts()

        ball_states = []
        for i in range(2):
            pos = self.data.qpos[self.ball_qpos_idx[i][:3]]
            quat_wxyz = self.data.qpos[self.ball_qpos_idx[i][3:7]]
            linvel = self.data.qvel[self.ball_qvel_idx[i][:3]]
            angvel = self.data.qvel[self.ball_qvel_idx[i][3:6]]
            ball_states.append({
                "pos": pos.copy(),
                "quat_xyzw": quat_wxyz_to_xyzw(quat_wxyz),
                "linvel": linvel.copy(),
                "angvel": angvel.copy(),
            })

        return self.obs_builder.build(
            hand_qpos=hand_qpos,
            prev_targets=self.prev_targets,
            fsr_contacts=fsr_contacts,
            ball_states=ball_states,
        )

    def _get_fsr_contacts(self) -> np.ndarray:
        """Binary contact detection for 16 FSR sensors."""
        contacts = np.zeros(16, dtype=np.float32)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            for si, gid in enumerate(self.fsr_geom_ids):
                if gid < 0:
                    continue
                if g1 == gid or g2 == gid:
                    contacts[si] = 1.0
        return contacts

    def get_hand_qpos(self) -> np.ndarray:
        return self.data.qpos[self.hand_qpos_idx].copy()

    def get_arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_idx].copy()

    def get_ball_pos(self, ball_idx: int) -> np.ndarray:
        return self.data.qpos[self.ball_qpos_idx[ball_idx][:3]].copy()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
