#!/bin/bash -x
#SBATCH -M lonepeak 
#SBATCH --account=lonepeak-gpu
#SBATCH --partition=lonepeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --exclusive
#SBATCH -t 0:05:00
echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
echo " " | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
module load cuda
nvcc -O3 -o symatb mm4_atb_main.cu mm4_atb_gpu.cu
./symatb 37 37 728271 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symatb 999 999 999 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symatb 1024 1024 1024 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symatb 4096 4096 64 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symatb 8192 8192 16 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symatb 16 16 419430 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
