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
nvcc -O3 -o symab mm4_ab_main.cu mm4_ab_gpu.cu
./symab 37 37 728271 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symab 999 999 999 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symab 1024 1024 1024 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symab 4096 4096 64 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symab 8192 8192 16 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
REM./symab 16 16 419430 | tee -a lonepeak_mm_gpu.$SLURM_JOB_ID\.log
