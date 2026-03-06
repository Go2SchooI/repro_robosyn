"""M3: Observation builder for the baoding task.

Constructs the 366-dim full_stack_baoding observation vector matching
the Isaac Gym environment exactly:
  - 85 dims per frame x 4 stacked frames = 340 dims
  - 13 dims x 2 balls privileged info = 26 dims
  Total = 366 dims

Hand joint order alignment:
  Isaac Gym orders hand DOFs by URDF child name sort: finger0, thumb, finger1, finger2
  → obs[6:22] = [joint_0..3, joint_12..15, joint_4..7, joint_8..11]
  MuJoCo uses HAND_JOINT_NAMES = joint_0..15. We reorder to match Isaac so the
  same obs indices mean the same physical joint for the policy.
"""

import numpy as np
from typing import List

from mujoco_sim.utils import unscale, SPIN_AXIS_Z

# Index permutation: MuJoCo hand_qpos[ISAAC_HAND_ORDER[k]] = Isaac obs[6+k]
# Isaac hand DOF order: finger0 (0-3), thumb (12-15), finger1 (4-7), finger2 (8-11)
ISAAC_HAND_ORDER = np.array([0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.intp)

# Inverse: policy output is in Isaac order; when applying to prev_targets (MuJoCo order),
# action_mujoco[6+j] = action_isaac[6 + MUJOCO_TO_ISAAC_ACTION[j]]
MUJOCO_TO_ISAAC_ACTION = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7], dtype=np.intp)

N_OBS_DIM = 85
N_STACK = 4
N_PRIV_PER_BALL = 13
N_BALLS = 2
OBS_TOTAL = N_OBS_DIM * N_STACK + N_PRIV_PER_BALL * N_BALLS  # 366
VEL_OBS_SCALE = 0.2


class ObservationBuilder:
    """Builds the full_stack_baoding observation vector from MuJoCo state.

    Frame layout (85 dims):
        [0:6]   arm joint positions (zeroed)
        [6:22]  hand joint positions (normalized to [-1,1])
        [22:29] zeroed padding (7 dims)
        [29:45] previous target positions (normalized to [-1,1], hand only)
        [45:61] FSR tactile sensors (binary, 16 dims)
        [61:85] spin axis repeated 8x = [0,0,1]*8 (24 dims)

    Privileged info per ball (13 dims):
        [0:3]   position (x,y,z)
        [3:7]   quaternion (x,y,z,w) -- Isaac Gym convention
        [7:10]  linear velocity
        [10:13] angular velocity * 0.2
    """

    def __init__(
        self,
        hand_lower_limits: np.ndarray,
        hand_upper_limits: np.ndarray,
        arm_lower_limits: np.ndarray,
        arm_upper_limits: np.ndarray,
        n_stack: int = N_STACK,
    ):
        self.hand_lower = hand_lower_limits  # (16,)
        self.hand_upper = hand_upper_limits  # (16,)
        self.n_stack = n_stack

        self.all_lower = np.concatenate([arm_lower_limits, hand_lower_limits])
        self.all_upper = np.concatenate([arm_upper_limits, hand_upper_limits])

        self.obs_history = np.zeros(N_OBS_DIM * n_stack, dtype=np.float32)
        self.init_stack = True

    def reset(self):
        """Clear frame history on episode reset."""
        self.obs_history[:] = 0.0
        self.init_stack = True

    def build(
        self,
        hand_qpos: np.ndarray,
        prev_targets: np.ndarray,
        fsr_contacts: np.ndarray,
        ball_states: List[dict],
    ) -> np.ndarray:
        """Build the full 366-dim observation.

        Args:
            hand_qpos: (16,) current hand joint positions
            prev_targets: (22,) previous target positions (arm+hand)
            fsr_contacts: (16,) binary contact readings
            ball_states: list of 2 dicts, each with keys:
                'pos': (3,), 'quat_xyzw': (4,), 'linvel': (3,), 'angvel': (3,)

        Returns:
            obs: (366,) float32 array
        """
        frame = self._build_frame(hand_qpos, prev_targets, fsr_contacts)

        if self.init_stack:
            self.obs_history = np.tile(frame, self.n_stack)
            self.init_stack = False
        else:
            self.obs_history = np.concatenate(
                [frame, self.obs_history[:-N_OBS_DIM]]
            )

        priv = self._build_privileged(ball_states)
        obs = np.concatenate([self.obs_history, priv]).astype(np.float32)
        assert obs.shape[0] == OBS_TOTAL, f"Expected {OBS_TOTAL}, got {obs.shape[0]}"
        return obs

    def _build_frame(
        self,
        hand_qpos: np.ndarray,
        prev_targets: np.ndarray,
        fsr_contacts: np.ndarray,
    ) -> np.ndarray:
        frame = np.zeros(N_OBS_DIM, dtype=np.float32)

        all_qpos = np.zeros(22)
        all_qpos[6:22] = hand_qpos
        scaled_qpos = unscale(all_qpos, self.all_lower, self.all_upper)
        frame[0:6] = 0.0
        # Reorder hand to Isaac DOF order (finger0, thumb, finger1, finger2)
        frame[6:22] = scaled_qpos[6:22][ISAAC_HAND_ORDER]

        frame[22:29] = 0.0

        scaled_targets = unscale(prev_targets, self.all_lower, self.all_upper)
        frame[29:45] = scaled_targets[6:22][ISAAC_HAND_ORDER]

        frame[45:61] = fsr_contacts[:16]

        axis_rep = np.tile(SPIN_AXIS_Z, 8)  # (24,)
        frame[61:85] = axis_rep

        return frame

    def _build_privileged(self, ball_states: List[dict]) -> np.ndarray:
        """Build 26-dim privileged observation for 2 balls."""
        priv = np.zeros(N_PRIV_PER_BALL * N_BALLS, dtype=np.float32)
        for i, bs in enumerate(ball_states):
            offset = i * N_PRIV_PER_BALL
            priv[offset:offset + 3] = bs["pos"]
            priv[offset + 3:offset + 7] = bs["quat_xyzw"]
            priv[offset + 7:offset + 10] = bs["linvel"]
            priv[offset + 10:offset + 13] = bs["angvel"] * VEL_OBS_SCALE
        return priv
