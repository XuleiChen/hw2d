#!/bin/bash
#SBATCH --job-name=hw2d_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/xc2695/hw2d_fourierflow/hw2d_training.out

export WANDB_MODE=online
# 清空模块
module purge

# 切换到项目目录
cd /scratch/xc2695/hw2d_fourierflow/fourierflow

# 使用 Singularity 执行训练任务
singularity exec --nv \
--overlay /scratch/xc2695/hw2d_singularity/overlay-50G-10M.ext3:ro \
/scratch/xc2695/hw2d_singularity/cuda_11.6.2-cudnn8-devel-centos7.sif \
/bin/bash -c "source /ext3/env.sh; \
python -c 'import wandb; wandb.login(key=\"5df940dcfe91a3c355972f6dfa67dc64b390baac\")';export PYTHONPATH='/scratch/xc2695/hw2d_fourierflow/fourierflow:$PYTHONPATH';cd /scratch/xc2695/hw2d_fourierflow/fourierflow ;fourierflow train --trial 4 experiments/hw2d/config.yaml"


