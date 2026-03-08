scripts/teacher_baoding.sh 0

# 仅 test（不录轨迹）
bash scripts/teacher_baoding.sh 0 test=True checkpoint=/home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-03-08_22-38-18-83810/nn/baoding.pth headless=False task.env.numEnvs=16

# Test + 记录 env0 灵巧手关节期望到 CSV（文件名按时间生成：runs/env0_trajectory_YYYYMMDD_HHMMSS.csv）
bash scripts/teacher_baoding.sh 0 test=True checkpoint=/home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-03-08_00-55-05-83810/nn/last_baoding_ep_2700_rew_1234.712.pth headless=False task.env.numEnvs=16 task.env.recordEnv0TrajectoryCsv=runs

mkdir -p /home/jizexian/dexhand/in-hand-rotation/.mujoco

tar -xzf /home/jizexian/dexhand/in-hand-rotation/mujoco_sim/mujoco210-linux-x86_64.tar.gz -C /home/jizexian/dexhand/in-hand-rotation/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jizexian/dexhand/in-hand-rotation/.mujoco/

cd /home/jizexian/dexhand/in-hand-rotation/.mujoco/mujoco210/bin


./compile /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.urdf /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

./simulate /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

# MuJoCo 策略推理
python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --checkpoint /home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-03-08_22-38-18-83810/nn/baoding.pth --slow 2

python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --checkpoint /home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-03-08_00-55-05-83810/nn/last_baoding_ep_2700_rew_1234.712.pth --slow 20

# MuJoCo 离线轨迹跟踪（读 CSV，不加载策略，用于 sim2sim 环境一致性验证）
# 注意：CSV 必须用修复后的 Isaac 代码录制（header 包含真实 DOF 名称），旧 CSV 的 header 有误不可用
python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --trajectory-csv /home/jizexian/dexhand/in-hand-rotation/runs/env0_trajectory_20260308_190150.csv --slow 10

python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --trajectory-csv /home/jizexian/dexhand/in-hand-rotation/runs/env0_trajectory_20260306_022644.csv --slow 5


# Isaac Gym DOF 排序说明：
# Isaac Gym 按 URDF 分支节点的子 joint 名称的字典序排列，因此 Allegro 手的 hand DOF 顺序为：
#   finger0 (joint_0-3.0), thumb (joint_12-15.0), finger1 (joint_4-7.0), finger2 (joint_8-11.0)
# 而非 joint_0.0 ~ joint_15.0 的数值顺序。MuJoCo replay 代码会根据 CSV header 自动重映射。

# Obs 对齐说明（Gym vs MuJoCo）：
# - 单帧 85 维：[0:6] 臂零, [6:22] 手关节(需与 Isaac 顺序一致), [22:29] 零, [29:45] 上一步 target 手, [45:61] FSR, [61:85] spin_axis×8
# - 手顺序：MuJoCo 构建 obs 时已按 ISAAC_HAND_ORDER 重排 [6:22] 与 [29:45]，与 Gym 一致
# - 策略推理：policy 输出 action 为 Isaac 顺序，step() 内用 MUJOCO_TO_ISAAC_ACTION 重排为 MuJoCo 顺序再与 prev_targets 相加
# - FSR 顺序：两边 contact_sensor_names / FSR_GEOM_NAMES 相同，[45:61] 对齐
# - 球 privileged：pos(3)+quat_xyzw(4)+linvel(3)+angvel*0.2(3)，quat 与 vel_obs_scale 已对齐

# ---------- obs[45:61] FSR 从仿真到 RL 的流程 ----------
#
# 【Isaac Gym】
# 1) 仿真数据来源
#    - gym.acquire_net_contact_force_tensor(sim) → 每步 refresh_net_contact_force_tensor(sim)
#    - contact_tensor: (num_envs, num_bodies*3)，按 rigid body 顺序，每 body 一个 3D 力向量 (N)
#    - reshape 为 (num_envs, 49, 3)，49=当前场景 rigid body 数
# 2) 取 16 个 FSR 对应 body 的力
#    - sensor_handle_indices = find_actor_rigid_body_handle(contact_sensor_names)，顺序与 contact_sensor_names 一致
#    - contacts = contact_tensor[:, sensor_handle_indices, :]  # (N_env, 16, 3)
# 3) 标量 + 阈值二值化
#    - contacts = norm(contacts, dim=-1)  # 每个 FSR 的力大小
#    - contact_thresh：每个 env、每个 sensor 在 reset 时随机 [1.0, 1.0+sensorThresh]，默认 sensorThresh=1.0 → [1, 2]
#    - contacts = (contacts >= contact_thresh) ? 1.0 : 0.0
# 4) 域随机化 / 仿真现实化
#    - 延迟：以概率 latency（默认 0.2）保留上一帧 last_contacts，否则用当前 contacts → last_contacts 更新
#    - 噪声：mask = rand；若 mask < sensorNoise（默认 0.1）则置 0，否则保留；只对 last_contacts>0.1 的位置乘 mask
#    - sensed_contacts = 上述结果；若 disableSet 启用则部分 sensor 置 0
# 5) 写入 obs
#    - last_obs_buf[:, 45:61] = sensed_contacts
#
# 【MuJoCo】
# 1) 仿真数据来源
#    - 每步 mj_step 后，data.contact 里有本步所有接触对 (geom1, geom2, ...)
#    - 无“每 body 合力”接口，只能用接触对判断“某 geom 是否参与接触”
# 2) 取 16 个 FSR 对应 geom 的接触
#    - fsr_geom_ids = 按 FSR_GEOM_NAMES 顺序的 geom ID
#    - 遍历 data.contact，若 contact.geom1 或 geom2 等于某 FSR 的 gid，则该 FSR contacts[si]=1.0
# 3) 无阈值、无延迟、无噪声
#    - 直接二值：有接触=1，无接触=0
# 4) 写入 obs
#    - frame[45:61] = fsr_contacts[:16]（ObservationBuilder 中）
#
# 【差异小结】
# - Isaac：连续力 → 每 env 随机阈值 [1,2] N 二值化 → 延迟(0.2) + 噪声(0.1) → obs
# - MuJoCo：接触存在性二值，无阈值/延迟/噪声；若要做 sim2sim 一致，可在 MuJoCo 侧加类似阈值与（可选）延迟/噪声。

# ---------- 观测计算代码位置（方便对照） ----------
#
# 【Isaac Gym】obs_type=full_stack_baoding 时
#   - 入口：compute_observations() 约 1025 行 → 根据 obs_type 调 compute_contact_observations('fsbd')
#   - 单帧 + 栈 + 球：compute_contact_observations(self, mode='fsbd')，约 1425–1511 行
#     - last_obs_buf 单帧 85 维：1090–1135（unscale 手/臂、FSR、spin_axis、prev_targets、栈更新）
#     - 接触处理（contact_tensor → sensed_contacts）：1456–1480
#     - obj_buf 与 obs_buf 拼接：1508–1511
#   - 数据来源：1028–1039 行 refresh 各类 tensor，object_pose/object_linvel/object_angvel 来自 root_state_tensor[object_indices]
#   - 传感器顺序：contact_sensor_names 约 278 行；sensor_handle_indices 约 828 行
#
# 【MuJoCo】
#   - 入口：env._get_obs()，mujoco_sim/env.py 约 269–299 行
#   - 单帧 + 栈 + 球：mujoco_sim/observations.py
#     - ObservationBuilder.build() 约 77–108 行（拼 frame 栈 + privileged）
#     - _build_frame() 约 110–135 行（85 维：臂零、手重排、prev_target 重排、FSR、spin_axis）
#     - _build_privileged() 约 137–146 行（两球各 13 维：pos, quat_xyzw, linvel, angvel*0.2）
#   - FSR：env._get_fsr_contacts()，env.py 约 301–312 行
#   - 手/球顺序常量：observations.py 的 ISAAC_HAND_ORDER、MUJOCO_TO_ISAAC_ACTION；utils.py 的 FSR_GEOM_NAMES

# ---------- 对齐检查清单（任务仍失败时可逐项查） ----------
# [x] 手关节 obs[6:22]、obs[29:45] 顺序 → ISAAC_HAND_ORDER 重排
# [x] 策略 action 应用顺序 → step() 内 MUJOCO_TO_ISAAC_ACTION 重排
# [x] 单帧 85 维布局、栈 4 帧、privileged 26 维 (13×2 球)
# [x] 球 quat：MuJoCo wxyz→xyzw，vel_obs_scale=0.2
# [x] FSR 传感器名称顺序一致
# [ ] FSR 语义差异：Isaac 力阈值+延迟+噪声，MuJoCo 仅“是否接触”二值 → 可能导致策略在 MuJoCo 下看到不同 contact 分布
# [ ] 初始状态：两球/手的 reset 位置、spin_axis 是否与 Isaac 一致
# [ ] 物理/控制：PD 增益、substeps、control_freq_inv、关节限位、摩擦等

# ---------- 关节 range/limits 对齐（两边顺序不一致时的处理） ----------
#
# 【顺序】
# - Isaac：arm_hand_dof_lower/upper_limits 来自 get_asset_dof_properties(asset)，顺序 = Asset DOF 顺序（与 arm_hand_dof_pos 一致）。手部为 init 里 hand_qpos_init_override 的键顺序：joint_0,1,2,3, 12,13,14,15, 4..11（即 finger0, thumb, finger1, finger2）。
# - MuJoCo：HAND_JOINT_NAMES = joint_0.0..joint_15.0（数值顺序）；ARM 为 joint1..6。env.py 中 HAND_LOWER/UPPER_LIMITS、ALL_LOWER/UPPER 均为该顺序。
#
# 【使用方式】
# - Isaac：unscale(dof_pos, lower, upper)、scale(actions, lower, upper)、clamp(targets, lower, upper) 全程用同一索引 → limits 与 state/action 同序，无顺序问题。
# - MuJoCo 观测：hand_qpos、prev_targets 为 MuJoCo 序；用 all_lower/all_upper（MuJoCo 序）做 unscale，得到 scaled 再按 ISAAC_HAND_ORDER 重排写入 frame[6:22]、frame[29:45] → 每个关节用到的 limit 与自身一致，仅输出顺序变为 Isaac 序。
# - MuJoCo 控制：step() 中 actions 为 Isaac 序，经 MUJOCO_TO_ISAAC_ACTION 转为 MuJoCo 序后与 prev_targets 相加；clip(targets, ALL_LOWER, ALL_UPPER) 时 targets 与 ALL_* 均为 MuJoCo 序，正确。step_from_targets() 的 targets 由 load_trajectory_csv 重排为 MuJoCo 序后传入，clip 亦正确。
#
# 【数值】
# - 臂：URDF (tilted) 与 env.py 的 ARM_LOWER/UPPER 一致（如 0, 0.673, -0.91601, 3.1416, 2.263, -1.56901 等）。
# - 手：URDF 与 env.py 的 HAND_LOWER/UPPER（按 joint_0..15）一致：finger0/1/2 各 (-0.47,0.47), (-0.196,1.61), (-0.174,1.709), (-0.227,1.618)；拇指 (0.70,1.396), (0.3,1.163), (-0.189,1.644), (-0.162,1.719)。XML 中 range= 与上述一致。
#
# 结论：两边关节 range 数值一致；MuJoCo 侧 limits 按自身 DOF 顺序使用，观测/控制处已按 ISAAC_HAND_ORDER 与 MUJOCO_TO_ISAAC_ACTION 处理顺序差异，无需再对 limits 做重排。

# ---------- API 手册调研：两边变量与坐标系是否对齐（仅调研，未改代码） ----------
#
# 一、Isaac Gym（来源：Tensor API 文档 + PhysX 文档 + 社区）
#
# 1. Actor Root State Tensor (root_state_tensor / object_pose 等)
#    - 布局：13 float/actor → pos(3), quat(4), linvel(3), angvel(3)
#    - 坐标系：均为世界系（world frame）。位置、朝向、线速度、角速度均在世界坐标系下。
#    - 四元数顺序：PhysX 使用 (x, y, z, w)；Isaac 暴露的 root_state_tensor [3:7] 与 PhysX 一致，即 xyzw。
#    - 文档依据：Tensor API 页 “same layout as GymRigidBodyState”；社区/检索结果明确 position/velocity 为 world frame。
#
# 2. Net Contact Force Tensor (contact_tensor → FSR 观测)
#    - 含义：每个 rigid body 上受到的净接触力（3D 向量）。
#    - 坐标系：文档写明 “Contact forces are measured in the **world frame**”。
#    - 注意：与 mesh 碰撞时存在已知的报告不准问题；contact_collection 与 substeps 会影响数值。
#
# 3. DOF State (关节位置/速度)
#    - 关节位置：弧度（revolute）或米（prismatic）；顺序与 get_actor_dof_states 一致，按 asset 的 DOF 顺序。
#    - 关节速度：弧度/秒或米/秒；与 DOF 顺序一致。未在文档中写明“相对哪一坐标系”，对 revolute 而言通常就是关节角速度标量。
#
# 二、MuJoCo（来源：API Reference + 论坛 + 检索）
#
# 1. Free joint（球体等）qpos / qvel (mjData)
#    - qpos（7）：pos(3) 世界系位置；quat(4) 世界系朝向，格式为 **(w, x, y, z)**，与 Isaac 的 xyzw 不同。
#    - qvel（6）：前 3 为线速度（世界系），后 3 为角速度（世界系）。与 Isaac 一致均为世界系。
#    - 结论：位置、线速度、角速度两边都是世界系；仅四元数存储顺序不同，需在 MuJoCo→obs 时做 wxyz→xyzw 转换（当前 utils 已做）。
#
# 2. 接触 (data.contact)
#    - 无“每 body 合力”接口；只有接触对列表 (geom1, geom2, …)。当前 MuJoCo 侧用“某 FSR geom 是否出现在任意 contact 对”做二值，与 Isaac 的“世界系合力→标量→阈值”在语义与数值上均不完全一致。
#
# 三、对齐结论摘要
#
# - 位置 (pos)：两边均为世界系 → 对齐。
# - 四元数 (quat)：Isaac xyzw，MuJoCo wxyz；obs 中已统一为 xyzw（MuJoCo 侧转换）→ 对齐。
# - 线速度 (linvel)：两边均为世界系 → 对齐。
# - 角速度 (angvel)：两边均为世界系 → 对齐。
# - 接触力 (FSR)：Isaac 为世界系 3D 力→标量→阈值/延迟/噪声；MuJoCo 为“是否接触”二值，无坐标系问题但语义与数值不一致。
# - 关节 DOF：顺序已在别处通过 ISAAC_HAND_ORDER / CSV header 等处理；单位与含义一致（弧度/弧度每秒）。
#
# 四、建议
#
# - 若任务仍失败，可重点排查：① FSR 语义差异（力 vs 存在性、阈值/延迟/噪声）；② 初始状态与 spin_axis；③ 物理/控制参数；④ 两边重力/up 轴是否一致（均为 Z-up 则无问题）。