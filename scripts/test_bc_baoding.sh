GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train_distillation.py headless=False \
task.env.legacy_obs=False distill.bc_training=test \
task.env.objSet=ball task.env.is_distillation=True \
task=AllegroArmMOAR task.env.numEnvs=64 \
train.params.config.minibatch_size=1024 \
train.params.config.central_value_config.minibatch_size=1024 \
task.env.observationType=full_stack_baoding \
distill.ablation_mode=multi-modality-plus \
distill.learn.max_iterations=1000 \
wandb_activate=False \
${EXTRA_ARGS}
