#!/bin/bash
#SBATCH --job-name=zipllm
# SBATCH --output=srun_logs/%j-zipllm-llama2-7b-finetune.out
# SBATCH --error=srun_logs/%j-zipllm-llama2-7b-finetune.err
#SBATCH --output=srun_logs/%j-shearedllama-finetune.out
#SBATCH --error=srun_logs/%j-shearedllama-finetune.err

#number of CPUs to be used
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

#Define the number of hours the job should run.
#SBATCH --time=80:00:00

#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=1000G
#SBATCH --no-requeue

#Define the "gpu" partition for GPU-accelerated jobs
#SBATCH --partition=gpu100
#Define the number of GPUs used by your job
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
# SBATCH --exclude=gpu[268,269,271]
#SBATCH --ntasks-per-node=1
# SBATCH --export=NONE

#load an MPI module with SLURM support or a software module
module load openmpi/4.1.4
#
#Set the number of OpenMP threads per MPI task to SLURM internal variable value
export OMP_NUM_THREADS=8

LOG_DIR=./srun_logs

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)

export MASTER_ADDR=$master_addr
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * 8 ))
export MASTER_PORT=$(( 10000 + $RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

yaml_file=./scripts/train/yamls/pretrain/llama2-7b.yaml
# save_folder=./srun_logs/llama2-7b-20kcali-1.5x-finetune-webedu_lr_1e4_batch1024_4096_1000ba_test
save_folder=./srun_logs/zipllm_from_1.5_openorca_2x_20B_finetune_kd

source /nfs/scistore19/alistgrp/stang/miniconda3/bin/activate llama

srun /usr/bin/nvidia-smi

if [[ $num_nodes == 1 ]]; then composer ./scripts/train/train.py $yaml_file save_folder=$save_folder; 
else srun --output=$LOG_DIR/%x-%j-%n.out bash ./srun_launch.sh $yaml_file save_folder=$save_folder; fi