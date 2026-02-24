bash scripts/teacher_baoding.sh 0 test=True checkpoint=/home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-02-23_01-48-25-83810/nn/last_baoding_ep_2700_rew_1187.106.pth headless=False task.env.numEnvs=16

mkdir -p /home/jizexian/dexhand/in-hand-rotation/.mujoco

tar -xzf /home/jizexian/dexhand/in-hand-rotation/mujoco_sim/mujoco210-linux-x86_64.tar.gz -C /home/jizexian/dexhand/in-hand-rotation/.mujoco/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jizexian/dexhand/in-hand-rotation/.mujoco/

cd /home/jizexian/dexhand/in-hand-rotation/.mujoco/mujoco210/bin


./compile /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.urdf /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

./simulate /home/jizexian/dexhand/in-hand-rotation/assets/urdf/xarm6/xarm6_allegro_right_fsr_2023_thin_tilted.xml

python -m mujoco_sim.run --xml mujoco_sim/assets/allegro_baoding.xml --checkpoint /home/jizexian/dexhand/in-hand-rotation/runs/baoding/baodingS1.0_C0.0_M0.02026-02-23_14-41-32-83810/nn/last_baoding_ep_4200_rew_1296.3552.pth