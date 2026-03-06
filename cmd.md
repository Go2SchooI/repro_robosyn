# 仅 test（不录轨迹）
bash scripts/teacher_baoding.sh 0 test=True checkpoint=/home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-02-23_01-48-25-83810/nn/last_baoding_ep_2700_rew_1187.106.pth headless=False task.env.numEnvs=16

# Test + 记录 env0 灵巧手关节期望到 CSV（文件名按时间生成：runs/env0_trajectory_YYYYMMDD_HHMMSS.csv）
bash scripts/teacher_baoding.sh 0 test=True checkpoint=/home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-02-23_14-41-32-83810/nn/last_baoding_ep_4200_rew_1296.3552.pth headless=False task.env.numEnvs=16 task.env.recordEnv0TrajectoryCsv=runs

mkdir -p /home/jizexian/dexhand/in-hand-rotation/.mujoco

tar -xzf /home/jizexian/dexhand/in-hand-rotation/mujoco_sim/mujoco210-linux-x86_64.tar.gz -C /home/jizexian/dexhand/in-hand-rotation/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jizexian/dexhand/in-hand-rotation/.mujoco/

cd /home/jizexian/dexhand/in-hand-rotation/.mujoco/mujoco210/bin


./compile /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.urdf /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

./simulate /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

# MuJoCo 策略推理
python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --checkpoint /home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-02-23_14-41-32-83810/nn/last_baoding_ep_4200_rew_1296.3552.pth

# MuJoCo 离线轨迹跟踪（读 CSV，不加载策略，用于 sim2sim 环境一致性验证）
# 注意：CSV 必须用修复后的 Isaac 代码录制（header 包含真实 DOF 名称），旧 CSV 的 header 有误不可用
python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --trajectory-csv /home/jizexian/dexhand/in-hand-rotation/runs/env0_trajectory_20260306_022644.csv


# Isaac Gym DOF 排序说明：
# Isaac Gym 按 URDF 分支节点的子 joint 名称的字典序排列，因此 Allegro 手的 hand DOF 顺序为：
#   finger0 (joint_0-3.0), thumb (joint_12-15.0), finger1 (joint_4-7.0), finger2 (joint_8-11.0)
# 而非 joint_0.0 ~ joint_15.0 的数值顺序。MuJoCo replay 代码会根据 CSV header 自动重映射。

# Obs 对齐说明（Gym vs MuJoCo）：
# - 单帧 85 维：[0:6] 臂零, [6:22] 手关节(需与 Isaac 顺序一致), [22:29] 零, [29:45] 上一步 target 手, [45:61] FSR, [61:85] spin_axis×8
# - 手顺序：MuJoCo 构建 obs 时已按 ISAAC_HAND_ORDER 重排 [6:22] 与 [29:45]，与 Gym 一致
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