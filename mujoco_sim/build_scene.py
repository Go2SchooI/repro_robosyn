"""M1: Build MuJoCo MJCF scene from the URDF assets.

Cleans the tilted URDF (removes inline # comments, remaps mesh extensions),
compiles via MuJoCo, then adds two baoding balls and finger actuators.
Outputs mujoco_sim/assets/allegro_baoding.xml.
"""

import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = os.path.join(
    REPO_ROOT, "assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.urdf"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "allegro_baoding.xml")

HAND_JOINT_NAMES = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
]

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

BALL1_INIT_POS = "0.63 0.01 0.25"
BALL2_INIT_POS = "0.63 -0.015 0.25"


def _clean_urdf(urdf_text: str, urdf_dir: str) -> str:
    """Remove inline # comments, remap mesh paths, and fix duplicate inertials."""
    cleaned = re.sub(r'#[^"<>\n]*', '', urdf_text)

    def _replace_mesh(match):
        original = match.group(1)
        full_path = os.path.join(urdf_dir, original)
        base, ext = os.path.splitext(full_path)
        obj_candidate = base + ".obj"
        if not os.path.exists(full_path) and os.path.exists(obj_candidate):
            new_name = os.path.splitext(original)[0] + ".obj"
            return f'filename="{new_name}"'
        return match.group(0)

    cleaned = re.sub(r'filename="([^"]+)"', _replace_mesh, cleaned)

    # Remove duplicate <inertial> blocks within <link> elements.
    # Keep the first <inertial>...</inertial> and remove subsequent ones.
    def _remove_dup_inertials(text):
        result_lines = []
        in_link = False
        link_has_inertial = False
        skip_inertial = False
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('<link ') or stripped.startswith('<link>'):
                in_link = True
                link_has_inertial = False
            if stripped.startswith('</link>'):
                in_link = False
                link_has_inertial = False
            if in_link and '<inertial' in stripped and not skip_inertial:
                if link_has_inertial:
                    skip_inertial = True
                    continue
                link_has_inertial = True
            if skip_inertial:
                if '</inertial>' in stripped:
                    skip_inertial = False
                continue
            result_lines.append(line)
        return '\n'.join(result_lines)

    cleaned = _remove_dup_inertials(cleaned)
    return cleaned


def _create_stl_symlinks(mesh_dir: str):
    """Create .stl/.STL symlinks pointing to .obj files where the .stl is missing."""
    for obj_file in os.listdir(mesh_dir):
        if not obj_file.endswith(".obj"):
            continue
        base = os.path.splitext(obj_file)[0]
        for ext in [".stl", ".STL"]:
            stl_file = base + ext
            stl_path = os.path.join(mesh_dir, stl_file)
            if not os.path.exists(stl_path):
                os.symlink(
                    os.path.join(mesh_dir, obj_file),
                    stl_path,
                )


def _compile_urdf(urdf_path: str) -> mujoco.MjModel:
    """Load URDF via MuJoCo's MjSpec API with balanceinertia enabled.

    MuJoCo strips directory prefixes from URDF mesh filenames, so all
    mesh files are symlinked into a flat temp directory alongside the URDF.
    """
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    with open(urdf_path, "r") as f:
        urdf_text = f.read()

    urdf_text = _clean_urdf(urdf_text, urdf_dir)

    tmp_dir = tempfile.mkdtemp(prefix="mujoco_urdf_")
    cleaned_path = os.path.join(tmp_dir, "robot.urdf")
    with open(cleaned_path, "w") as f:
        f.write(urdf_text)

    mesh_dirs = [
        os.path.join(urdf_dir, "meshes"),
        os.path.normpath(os.path.join(
            urdf_dir, "..", "allegro_hand_description", "meshes"
        )),
    ]
    for mdir in mesh_dirs:
        if not os.path.isdir(mdir):
            continue
        for fname in os.listdir(mdir):
            src = os.path.join(mdir, fname)
            dst = os.path.join(tmp_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    try:
        spec = mujoco.MjSpec()
        spec.from_file(cleaned_path)
        spec.balanceinertia = True
        spec.boundmass = 0.001
        spec.boundinertia = 1e-6
        model = spec.compile()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return model


def _save_and_patch_xml(model: mujoco.MjModel) -> str:
    """Export MjModel to XML string, then patch it with balls/actuators."""
    mujoco.mj_saveLastXML("/tmp/_allegro_base.xml", model)

    with open("/tmp/_allegro_base.xml", "r") as f:
        xml_str = f.read()

    tree = ET.parse("/tmp/_allegro_base.xml")
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.008335")  # Isaac physics_dt = sim.dt/substeps = 0.01667/2
    option.set("gravity", "0 0 -9.81")
    option.set("cone", "pyramidal")
    option.set("solver", "Newton")
    option.set("iterations", "10")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    asset_dir = os.path.join(REPO_ROOT, "assets", "urdf")
    compiler.set("meshdir", asset_dir)

    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    default_geom = ET.SubElement(default, "geom")
    default_geom.set("condim", "4")
    default_geom.set("friction", "1.0 0.005 0.001")
    default_geom.set("solref", "0.004 1")
    default_geom.set("solimp", "0.95 0.99 0.001")

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    ball1 = ET.SubElement(worldbody, "body")
    ball1.set("name", "ball1")
    ball1.set("pos", BALL1_INIT_POS)
    fj1 = ET.SubElement(ball1, "freejoint")
    fj1.set("name", "ball1_joint")
    g1_vis = ET.SubElement(ball1, "geom")
    g1_vis.set("name", "ball1_visual")
    g1_vis.set("type", "sphere")
    g1_vis.set("size", "0.022")
    g1_vis.set("rgba", "0.9 0.3 0.3 1")
    g1_vis.set("contype", "0")
    g1_vis.set("conaffinity", "0")
    g1_col = ET.SubElement(ball1, "geom")
    g1_col.set("name", "ball1_collision")
    g1_col.set("type", "sphere")
    g1_col.set("size", "0.022")
    g1_col.set("mass", "0.25")
    g1_col.set("condim", "4")
    g1_col.set("friction", "1.0 0.005 0.001")

    ball2 = ET.SubElement(worldbody, "body")
    ball2.set("name", "ball2")
    ball2.set("pos", BALL2_INIT_POS)
    fj2 = ET.SubElement(ball2, "freejoint")
    fj2.set("name", "ball2_joint")
    g2_vis = ET.SubElement(ball2, "geom")
    g2_vis.set("name", "ball2_visual")
    g2_vis.set("type", "sphere")
    g2_vis.set("size", "0.022")
    g2_vis.set("rgba", "0.3 0.3 0.9 1")
    g2_vis.set("contype", "0")
    g2_vis.set("conaffinity", "0")
    g2_col = ET.SubElement(ball2, "geom")
    g2_col.set("name", "ball2_collision")
    g2_col.set("type", "sphere")
    g2_col.set("size", "0.022")
    g2_col.set("mass", "0.25")
    g2_col.set("condim", "4")
    g2_col.set("friction", "1.0 0.005 0.001")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    for jname in HAND_JOINT_NAMES:
        motor = ET.SubElement(actuator, "motor")
        motor.set("name", f"act_{jname}")
        motor.set("joint", jname)
        motor.set("ctrlrange", "-20 20")
        motor.set("ctrllimited", "true")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for jname in ARM_JOINT_NAMES:
        jc = ET.SubElement(equality, "joint")
        jc.set("joint1", jname)
        jc.set("polycoef", "0 1 0 0 0")

    _set_joint_properties(root)

    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")

    return tree


def _set_joint_properties(root):
    """Set armature, damping, and other joint properties."""
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            jname = joint.get("name", "")
            if jname in ARM_JOINT_NAMES:
                joint.set("armature", "0.1")
                joint.set("damping", "100.0")
            elif jname in HAND_JOINT_NAMES:
                joint.set("armature", "0.1")
                joint.set("damping", "0.0")


def build():
    print("Compiling URDF with MuJoCo...")
    model = _compile_urdf(URDF_PATH)
    print(f"  Loaded model: {model.nq} qpos, {model.nv} qvel, {model.nbody} bodies")

    print("Patching XML with balls, actuators, and constraints...")
    tree = _save_and_patch_xml(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tree.write(OUTPUT_PATH, xml_declaration=True, encoding="unicode")

    test_model = mujoco.MjModel.from_xml_path(OUTPUT_PATH)
    test_data = mujoco.MjData(test_model)
    print(f"  Final model: {test_model.nq} qpos, {test_model.nv} qvel, "
          f"{test_model.nbody} bodies, {test_model.nu} actuators")
    print(f"  Ball 1 qpos indices: {test_model.jnt_qposadr[mujoco.mj_name2id(test_model, mujoco.mjtObj.mjOBJ_JOINT, 'ball1_joint')]}")
    print(f"  Ball 2 qpos indices: {test_model.jnt_qposadr[mujoco.mj_name2id(test_model, mujoco.mjtObj.mjOBJ_JOINT, 'ball2_joint')]}")

    print(f"\nJoint names and qpos addresses:")
    for i in range(test_model.njnt):
        name = mujoco.mj_id2name(test_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_adr = test_model.jnt_qposadr[i]
        print(f"  {name}: qpos[{qpos_adr}]")

    print(f"\nActuator names:")
    for i in range(test_model.nu):
        name = mujoco.mj_id2name(test_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {name}")

    print(f"\nScene saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build()
