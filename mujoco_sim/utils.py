import numpy as np


def quat_wxyz_to_xyzw(q):
    """MuJoCo (w,x,y,z) -> Isaac Gym (x,y,z,w)."""
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    """Isaac Gym (x,y,z,w) -> MuJoCo (w,x,y,z)."""
    return np.array([q[3], q[0], q[1], q[2]])


def unscale(x, lower, upper):
    """Map from [lower, upper] to [-1, 1]. Matches Isaac Gym's unscale()."""
    return (2.0 * x - upper - lower) / (upper - lower)


def scale(x, lower, upper):
    """Map from [-1, 1] to [lower, upper]. Matches Isaac Gym's scale()."""
    return (0.5 * (x + 1.0)) * (upper - lower) + lower


HAND_JOINT_NAMES = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
]

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

FSR_GEOM_NAMES = [
    "link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
    "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr",
    "link_9.0_fsr", "link_10.0_fsr", "link_11.0_tip_fsr",
    "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
    "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr",
]

SPIN_AXIS_Z = np.array([0.0, 0.0, 1.0])
