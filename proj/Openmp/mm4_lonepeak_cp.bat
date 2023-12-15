#!/bin/bash -x
#SBATCH -M lonepeak 
#SBATCH --account=lonepeak-gpu 
#SBATCH --partition=lonepeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
echo " " | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
echo "Par-No-UNR " | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
gcc -O3 -fopenmp -o sym original\ files/main.c
./sym 1024 1024 1024 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
REM./sym 37 37 728271 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
REM./sym 16 16 4194304 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
REM./sym 8192 8192 16 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
REM./sym 4096 4096 64 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
REM./sym 999 999 999 | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
