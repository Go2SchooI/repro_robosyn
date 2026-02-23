# Robot Synesthesia 仓库结构化报告

> 生成时间：2026-02-23 | 基于 commit: main 分支当前 HEAD

---

## 1. 项目一句话概述

**做什么：** 基于 Isaac Gym 仿真训练灵巧手（Allegro Hand + xArm6）的物体旋转操控策略——先用 privileged state 训练 teacher，再通过行为克隆蒸馏到仅用视觉+触觉的 student（ICRA 2024 论文 *Robot Synesthesia*）。

**不做什么：** 不涉及真机部署代码、不含 sim-to-real 迁移模块、不含传感器标定或硬件驱动。

---

## 2. 目录结构总览

```
in-hand-rotation/
├── isaacgymenvs/              # 核心：Isaac Gym 环境 + RL 训练入口
│   ├── __init__.py            # OmegaConf resolver 注册 + make() 工厂函数
│   ├── train.py               # ★ 主入口 1：Hydra + rl_games 训练 teacher
│   ├── train_distillation.py  # ★ 主入口 2：蒸馏/数据收集
│   ├── cfg/                   # Hydra 配置
│   │   ├── config.yaml            # teacher 训练总配置
│   │   ├── config_distill.yaml    # 蒸馏训练总配置
│   │   ├── task/AllegroArmMOAR.yaml  # 任务环境参数（奖励/DR/物理/传感器）
│   │   └── train/AllegroArmMOARPPO.yaml  # PPO 算法超参（网络/学习率/epoch）
│   ├── tasks/                 # Isaac Gym 任务定义
│   │   ├── base/vec_task.py           # VecTask 基类（gym 接口封装 + DR 基础设施）
│   │   └── allegro_arm_morb_axis.py   # ★ 核心环境（2109 行）：场景/观测/奖励/reset
│   ├── learning/              # 学习算法（在 rl_games 基础上定制）
│   │   ├── common_agent.py / common_player.py  # 通用 agent/player（next-obs tracking）
│   │   ├── amp_continuous.py      # AMP Agent（AMPAgent 继承 CommonAgent）
│   │   ├── amp_players.py         # AMP Player（评估用）
│   │   ├── amp_models.py          # AMP 模型包装
│   │   ├── amp_network_builder.py # AMP 网络（含 discriminator）
│   │   ├── amp_datasets.py        # AMP 数据集
│   │   ├── hrl_continuous.py      # 层级 RL（高层 latent + 低层 LLC）
│   │   ├── hrl_models.py          # HRL 模型包装
│   │   └── replay_buffer.py       # 经验回放缓冲区
│   └── utils/                 # 工具函数
│       ├── rlgames_utils.py       # RLGPUEnv 适配器 + AlgoObserver
│       ├── pc_utils.py            # 点云处理（归一化/采样）
│       ├── rotation3d.py          # 四元数/旋转运算（torch.jit 加速）
│       ├── dr_utils.py            # Domain Randomization 工具函数
│       ├── torch_jit_utils.py     # JIT 辅助函数
│       ├── reformat.py            # OmegaConf → dict 转换
│       └── utils.py               # seed 设置 / numpy 格式化
│
├── rl_games/                  # ★ rl_games 库的 vendored fork（本地修改版）
│   ├── torch_runner.py            # Runner 总调度（训练/评估分发）
│   ├── algos_torch/               # PyTorch RL 算法实现
│   │   ├── a2c_continuous.py          # A2C 连续动作（PPO 梯度计算）
│   │   ├── a2c_discrete.py           # A2C 离散动作
│   │   ├── network_builder.py        # 网络构建器（MLP/CNN/RNN）
│   │   ├── model_builder.py          # 模型注册工厂
│   │   ├── central_value.py          # Central Value Network
│   │   ├── players.py                # 评估 Player
│   │   ├── pointnets.py              # PointNet 编码器
│   │   ├── running_mean_std.py       # 运行均值/标准差
│   │   ├── sac_agent.py              # SAC（未使用）
│   │   └── torch_ext.py              # PyTorch 扩展工具
│   ├── common/                    # 公共基础设施
│   │   ├── a2c_common.py             # A2C 基类（训练循环/GAE/experience buffer）
│   │   ├── vecenv.py                 # 向量化环境接口
│   │   ├── env_configurations.py     # 环境注册表
│   │   ├── experience.py             # Experience Buffer
│   │   ├── player.py                 # BasePlayer
│   │   ├── schedulers.py             # 学习率调度器
│   │   ├── datasets.py               # PPO Dataset
│   │   └── ...                       # 其他辅助模块
│   ├── configs/                   # 原库自带配置（150+ yaml，大部分与本项目无关）
│   ├── envs/                      # 原库自带环境适配（Atari/SMAC 等，未使用）
│   ├── interfaces/                # 算法抽象接口
│   └── networks/                  # TCNN-MLP 等网络
│
├── distillation/              # Teacher→Student 蒸馏模块
│   ├── rl_pytorch/rl_pytorch/distill/
│   │   ├── distill_collect.py         # 用 teacher 收集 demonstration 数据
│   │   └── distill_bc_warmup.py       # 行为克隆训练 student
│   └── utils/
│       ├── config.py                  # CLI 参数解析（gymutil 风格）
│       └── process_distill.py         # 根据 bc_training 模式分发 collector/trainer
│
├── scripts/                   # 12 个 shell 启动脚本
│   ├── teacher_baoding.sh / teacher_cross.sh / teacher_axis.sh       # Teacher 训练
│   ├── teacher_baoding_visrl.sh / teacher_cross_visrl.sh / teacher_axis_visrl.sh  # Visual RL baseline
│   ├── collect_baoding.sh / collect_cross.sh / collect_axis.sh       # 数据收集
│   └── bc_baoding.sh / bc_cross.sh / bc_axis.sh                     # 行为克隆训练
│
├── assets/                    # 仿真资源文件
│   └── urdf/
│       ├── xarm6/                     # 机械臂+灵巧手 URDF（含 FSR 传感器）
│       └── objects/                   # 22 种操作物体 URDF（球/十字/方块/圆柱等）
│
├── pickle_utils.py            # gzip pickle 读写工具
├── readme.md                  # 项目说明（含训练/蒸馏命令）
├── install.md                 # 安装指南
├── cmd.md                     # 临时命令备忘（个人文件）
├── runs/                      # 实验输出（训练 checkpoint/config）
└── wandb/                     # Weights & Biases 日志
```

### Python 代码量分布

| 模块 | 行数 | 占比 |
|------|------|------|
| `rl_games/` | ~10,000 | 53% |
| `isaacgymenvs/` | ~7,100 | 37% |
| `distillation/` | ~1,240 | 7% |
| 根目录 (`pickle_utils.py`) | 32 | <1% |
| **合计** | **~19,000** | |

---

## 3. 关键入口与主流程

### 3.1 流程 A：Teacher 训练（PPO with privileged state）

```
scripts/teacher_baoding.sh 0
  └─ python isaacgymenvs/train.py  (Hydra: config.yaml + task/AllegroArmMOAR.yaml)
      │
      ├─ 1. isaacgymenvs/__init__.py::make()
      │      └─ tasks/__init__.py → AllegroArmMOAR(VecTask)  [环境实例化]
      │
      ├─ 2. 注册 rl_games 组件:
      │      vecenv.register('RLGPU', RLGPUEnv)
      │      algo_factory.register('amp_continuous', AMPAgent)
      │
      ├─ 3. rl_games/torch_runner.py::Runner.load() + reset()
      │      └─ 创建 A2CAgent (或 AMPAgent)
      │
      └─ 4. Runner.run({'train': True})
             └─ a2c_common.py::train()
                 ├─ play_steps() → 在 AllegroArmMOAR 中收集 rollout
                 ├─ calc_gradients() → PPO 更新
                 └─ 循环至 max_epochs
```

### 3.2 流程 B：数据收集 + 行为克隆

```
scripts/collect_baoding.sh 0 teacher_logdir=... teacher_resume=...
  └─ python isaacgymenvs/train_distillation.py  (Hydra: config_distill.yaml)
      │
      ├─ 1. 创建 vec_env（同上）
      │
      ├─ 2. process_distill.py::process_distill_trainer()
      │      ├─ bc_training="collect" → DistillCollector
      │      │    加载 teacher checkpoint → rollout → 保存 (obs, action) 数据
      │      │
      │      └─ bc_training="warmup" → DistillWarmUpTrainer
      │           加载离线数据 → BC loss 训练 student 网络
      │
      └─ 3. distiller.run(num_learning_iterations=...)
```

### 3.3 流程 C：Visual RL Baseline

```
scripts/teacher_baoding_visrl.sh 0
  └─ 与流程 A 相同，但使用:
     observationType=partial_stack_baoding (非 full_stack)
     ablation_mode=multi-modality-plus
     numEnvs=64 (需渲染点云，内存受限)
```

### 3.4 类继承链

```
VecTask (base/vec_task.py)
  └─ AllegroArmMOAR (tasks/allegro_arm_morb_axis.py)

ContinuousA2CBase (rl_games/common/a2c_common.py)
  └─ A2CAgent (rl_games/algos_torch/a2c_continuous.py)
      └─ CommonAgent (isaacgymenvs/learning/common_agent.py)
          ├─ AMPAgent (isaacgymenvs/learning/amp_continuous.py)
          └─ HRLAgent (isaacgymenvs/learning/hrl_continuous.py)
```

---

## 4. 关键数据结构/配置

### 4.1 配置入口一览

| 文件 | 角色 | 关键参数 |
|------|------|---------|
| `cfg/config.yaml` | Teacher 训练总控 | `task_name`, `seed`, `sim_device`, `wandb_*`, `test`, `checkpoint` |
| `cfg/config_distill.yaml` | 蒸馏总控 | 同上 + `distill.*`（bc_training, teacher_data_dir, ablation_mode, learn.*） |
| `cfg/task/AllegroArmMOAR.yaml` | 环境定义 | `env.*`（objSet, observationType, numEnvs, reward 系数, camera, sensor, DR 参数）; `sim.*`（PhysX 参数） |
| `cfg/train/AllegroArmMOARPPO.yaml` | 算法超参 | `network.*`（MLP units, pointnet）; `config.*`（lr, epochs, minibatch, horizon, central_value） |

### 4.2 观测模式 (`observationType`)

| 模式 | 维度 | 用途 |
|------|------|------|
| `full_stack` | 353 | Teacher（privileged state + 触觉） |
| `full_stack_baoding` | 366 | Teacher（baoding 双球版） |
| `full_stack_pointcloud` | 385 | Teacher（含点云嵌入） |
| `partial_stack` | 340 | Student / Visual RL |
| `partial_stack_baoding` | 340 | Student（baoding 版） |
| `partial_stack_pointcloud` | 340 | Student（含点云） |

每帧 85 维观测组成：关节位置 22 + 目标关节 23 + 触觉 16 + 旋转轴 24。堆叠 4 帧 = 340。Teacher 额外追加物体 pose 13 维。

### 4.3 消融模式 (`ablation_mode`)

| 模式 | 说明 |
|------|------|
| `multi-modality-plus` | Touch + Cam + Aug + Syn（完整版） |
| `aug` | Touch + Cam + Aug |
| `no-tactile` | 无触觉 |
| `no-pc` | 无点云（Teacher 默认） |
| `multi-modality` | 基础多模态 |

### 4.4 物体集合

| `objSet` | 物体 | 任务 |
|----------|------|------|
| `ball` | 2 × 半径 2.2cm 球 | Baoding 双球旋转 |
| `cross` | 5 种十字形 | Wheel-wrench 旋转 |
| `C` | 16 种不规则体 | 三轴旋转（curriculum） |

### 4.5 奖励组成

| 项 | 公式 | 默认系数 | 含义 |
|----|------|---------|------|
| `spin_reward` | `spin_coef × θ × 20` | 1.0 | 绕目标轴旋转角度 |
| `vel_reward` | `vel_coef × ‖linvel‖` | -0.1 | 惩罚物体线速度 |
| `distance_reward` | `0.1/(4d+0.02)` per fingertip | 0.1 | 指尖靠近物体 |
| `contact_reward` | `clip(Σcontacts, 0, 5)` | 0.0 | 鼓励多指接触 |
| `torque_penalty` | `Σ(τ²)` | -0.0003 | 力矩正则化 |
| `work_penalty` | `Σ(\|τ\|·\|dq/dt\|)` | -0.0003 | 做功正则化 |

### 4.6 域随机化（四层）

| 层级 | 随机化内容 | 触发时机 | 受 `randomize` 开关控制？ |
|------|-----------|---------|-------------------------|
| 创建时 | 物体类别/缩放/初始摩擦/初始质量 | 程序启动一次 | 否 |
| Episode Reset | 质量(0.2–0.6)、摩擦(0.2–3.0)、PD 增益(P:30–60, D:3–4.2)、传感器阈值 | 每次 env reset | **否（硬编码）** |
| Isaac Gym DR | 观测/动作噪声、重力扰动、关节刚度/限位 | 每 ≥1000 步 + reset | **是** |
| 每步噪声 | 触觉延迟(25%)/dropout(10%)、关节观测噪声(±0.06) | 每个 step | 否 |

---

## 5. 依赖与运行方式

### 5.1 安装

```bash
# 1. Conda 环境
conda create -n robosyn python=3.8
conda activate robosyn
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# 2. Isaac Gym Preview 4
pip install scipy imageio ninja
# 下载 IsaacGym_Preview_4_Package.tar.gz 并解压
cd isaacgym/python && pip install -e . --no-deps

# 3. 其他依赖
pip install hydra-core gym ray open3d numpy==1.20.3 tensorboardX tensorboard wandb
```

> **注意：没有 `requirements.txt` / `setup.py` / `pyproject.toml`。** 依赖信息完全散落在 `install.md` 中。

### 5.2 训练命令

```bash
# Teacher 训练 (baoding 双球旋转)
bash scripts/teacher_baoding.sh <GPU_ID>

# Teacher 训练 (wheel-wrench)
bash scripts/teacher_cross.sh <GPU_ID>

# Teacher 训练 (三轴旋转, 以 x 轴为例)
bash scripts/teacher_axis.sh 0 task.env.axis=x experiment=x-axis train.params.config.user_prefix=x-axis

# 数据收集
bash scripts/collect_baoding.sh <GPU_ID> teacher_logdir=<path> teacher_resume=<name>

# 行为克隆
bash scripts/bc_baoding.sh <GPU_ID> distill.teacher_data_dir=<path>

# 评估已有 checkpoint
bash scripts/teacher_baoding.sh 0 test=True checkpoint=<path> headless=False task.env.numEnvs=16
```

### 5.3 测试

**没有任何自动化测试**（无 pytest、无 CI/CD 配置文件）。

---

## 6. "需要整理的点"清单

### P0 — 阻碍正确运行或新人上手

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **缺少 `requirements.txt`** | 根目录 | 新人无法 `pip install -r` 一键装依赖 |
| 2 | **`runs/` 和 `wandb/` 未被 gitignore** | `.gitignore` 只有 3 行 | 训练产物/日志被 tracked，仓库膨胀 |
| 3 | **HRL 模块引用不存在的 `gen_amp*` 文件** | `hrl_continuous.py:49-51` | 使用 HRL 路径会立即 ImportError |
| 4 | **YAML 拼写错误 `truncate_grads: Trues`** | `AllegroArmMOARPPO.yaml:68` | 可能被当作 string 而非 bool |

### P1 — 代码质量/可维护性

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 5 | **`set_seed` / `set_np_formatting` 在 3 处重复定义** | `isaacgymenvs/utils/utils.py`, `distillation/utils/config.py`, `rl_games` 内部 | 改一处忘其他处 |
| 6 | **`train.py` 和 `train_distillation.py` 90% 重复** | 两个入口文件 | 几乎所有 env 注册和 wandb 初始化逻辑完全一致 |
| 7 | **`sys.path.append` hack** | `train.py:43`, `train_distillation.py:43` | 脆弱的路径操作，取决于 CWD |
| 8 | **`allegro_arm_morb_axis.py` 巨型文件 2109 行** | `tasks/` | 环境创建、奖励、观测、点云、reset、DR 全在一个文件中 |
| 9 | **`rl_games/` 整库 vendored 无版本标记** | `rl_games/` | 不知道基于哪个版本 fork，也不知道做了什么修改 |
| 10 | **`distillation/rl_pytorch/rl_pytorch/` 双层嵌套** | `distillation/` | 目录层级无意义冗余 |
| 11 | **`process_distill.py` 大量参数透传（30+ 个）** | `distillation/utils/` | 两个分支 80% 参数重复，应用 dataclass/config 对象 |
| 12 | **Shell 脚本间高度重复** | `scripts/*.sh` | 12 个脚本只在 3-4 个参数上不同 |
| 13 | **`distillation/utils/config.py` 包含 A1/Robot 等无关任务** | `distillation/utils/` | 从其他项目复制来的代码，本项目未使用 |
| 14 | **`rl_games/configs/` 包含 150+ 无关配置** | `rl_games/configs/` | Atari/SMAC/procgen 等与本项目无关 |
| 15 | **`pickle_utils.py` 放在根目录** | 根目录 | 应归入 utils/ |
| 16 | **`cmd.md` 临时文件** | 根目录 | 个人临时命令，不应入库 |

### P2 — 架构/长期演进

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 17 | **无自动化测试** | 全局 | 重构无安全网 |
| 18 | **无 `setup.py` / `pyproject.toml`** | 根目录 | 无法 `pip install -e .`，导入依赖 `sys.path` hack |
| 19 | **环境和算法耦合** | `common_agent.py` 直接引用 AMP 数据集 | 非 AMP 算法也被强制带入 AMP 依赖 |

---

## 7. 三阶段整理计划

### Phase 1: Quick Wins（1 小时内）

| 改动 | 范围 | 验收标准 |
|------|------|---------|
| 修复 `.gitignore`：添加 `runs/`, `wandb/`, `*.pth`, `demonstration-*/`, `videos/` | `.gitignore` | `git status` 不再显示 `runs/`, `wandb/` |
| 修复 `truncate_grads: Trues` → `True` | `AllegroArmMOARPPO.yaml:68` | YAML lint 通过 |
| 删除/gitignore `cmd.md` | 根目录 | 仓库中无个人临时文件 |
| 生成 `requirements.txt`（固定版本） | 根目录 | 新人 `pip install -r requirements.txt` 可安装所有依赖 |
| 将 `pickle_utils.py` 移入 `isaacgymenvs/utils/` | 根目录 → `isaacgymenvs/utils/` | 根目录只保留入口/文档/配置 |
| 在 README 中补充项目结构说明 | `readme.md` | 新人 5 分钟内能找到"代码在哪改" |

### Phase 2: 中等改动（1 天内）

| 改动 | 范围 | 验收标准 |
|------|------|---------|
| **合并 `train.py` / `train_distillation.py`**：提取公共逻辑到 `launch_common()`，两入口只保留差异分支 | `isaacgymenvs/train*.py` | 减少 ~120 行重复，功能不变 |
| **清理或隔离 vendored `rl_games/`**：添加 `rl_games/README.md` 标注 fork 版本和修改内容；删除 `configs/` 中无关 YAML | `rl_games/` | 只保留相关配置；有 CHANGELOG 说明修改 |
| **删除死代码**：`hrl_continuous.py` 中对 `gen_amp*` 的引用；`distillation/utils/config.py` 中的 A1/Robot 路径 | 多个文件 | 所有 import 可正常解析 |
| **统一 `set_seed` 实现**：保留 `isaacgymenvs/utils/utils.py` 一份，其余引用它 | 3 个文件 | `grep -r "def set_seed"` 只出现 1 次 |
| **合并 shell 脚本**：写一个 `scripts/run.sh <mode> <task> <gpu> [extra_args]` 模板替代 12 个脚本 | `scripts/` | 1 个脚本覆盖原有 12 种组合 |
| **添加 `pyproject.toml`**：使项目可 `pip install -e .` | 根目录 | 去掉 `sys.path.append` hack |
| **扁平化 distillation 目录**：`distillation/rl_pytorch/rl_pytorch/distill/` → `distillation/distill/` | `distillation/` | 目录深度减少 2 层 |

### Phase 3: 大改（多天）

| 改动 | 范围 | 验收标准 |
|------|------|---------|
| **拆分 `allegro_arm_morb_axis.py`（2109 行）**：分为 `env_setup.py`（资产加载/创建 sim）、`observations.py`（观测计算/点云）、`rewards.py`（奖励函数）、`allegro_env.py`（组装入口） | `isaacgymenvs/tasks/` | 每个文件 < 500 行，原有训练结果可复现 |
| **用 dataclass 重构 distillation 参数传递**：`process_distill.py` 的 30+ 参数 → `DistillConfig` dataclass | `distillation/` | 构造函数签名 < 10 个参数 |
| **添加集成测试**：冒烟测试（numEnvs=2 跑 10 步不报错）+ observation shape 断言 | `tests/` | `pytest tests/` 绿灯 |
| **评估是否升级/解耦 `rl_games`**：对比官方最新版本 diff，确认本地修改是否已被合入 | `rl_games/` | 明确记录"必须 vendor"或"可用 pip 官方版" |
| **解耦 AMP 依赖**：`CommonAgent` 不应默认依赖 AMP dataset；AMP 逻辑完全下沉到 `AMPAgent` | `isaacgymenvs/learning/` | `CommonAgent` 可独立于 AMP 使用 |

**建议起步顺序：** Phase 1 全做 → Phase 2 中的"合并 train.py" + "清理 rl_games" + "添加 pyproject.toml" → Phase 3 中的"拆分巨型文件"。每个阶段完成后跑一次完整的 teacher training 确认不回归。

---

## 附录 A：URDF 结构

### 使用的 URDF 文件

| 任务 | URDF |
|------|------|
| cross / C（轴旋转） | `xarm6_allegro_right_fsr_2023_thin.urdf` |
| ball（baoding 双球） | `xarm6_allegro_right_fsr_2023_thin_tilted.urdf`（手掌倾斜） |

### 运动链

```
world (fixed)
└── link_base                              ← xArm6 底座
    └── joint1 (revolute, 锁死)            ← xArm6 关节 1
        └── link1
            └── joint2 (revolute, 锁死)
                └── link2
                    └── joint3 (revolute, 锁死)
                        └── link3
                            └── joint4 (revolute, 锁死)
                                └── link4
                                    └── joint5 (revolute, 锁死)
                                        └── link5
                                            └── joint6 (revolute, 锁死)
                                                └── link6
                                                    └── gripper_base_joint (fixed)
                                                        └── base_link  ← Allegro Hand 掌根
                                                            ├── Finger 0 (食指): 4 DOF + FSR
                                                            ├── Finger 1 (中指): 4 DOF + FSR
                                                            ├── Finger 2 (无名指): 4 DOF + FSR
                                                            ├── Thumb (拇指): 4 DOF + FSR
                                                            └── palm + 4 个掌根 FSR
```

### DOF 总结（22 个）

| 编号 | 关节 | 实际运动范围 | 说明 |
|------|------|-------------|------|
| 0–5 | joint1–joint6 | ~0（锁死） | xArm6（URDF 限位差 ~0.00001 rad + 高阻尼 100.0） |
| 6–9 | joint_0.0–joint_3.0 | 正常 | 食指 4 DOF |
| 10–13 | joint_4.0–joint_7.0 | 正常 | 中指 4 DOF |
| 14–17 | joint_8.0–joint_11.0 | 正常 | 无名指 4 DOF |
| 18–21 | joint_12.0–joint_15.0 | 正常 | 拇指 4 DOF |

action 空间 22 维，前 6 维被物理约束吞掉，实际只控制 16 个手指关节。

### FSR 传感器（16 个）

每根手指 3 个（近端指节、中间指节、指尖）+ 4 个掌根传感器。代码通过接触力二值化后送入观测。
