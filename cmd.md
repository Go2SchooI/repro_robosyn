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