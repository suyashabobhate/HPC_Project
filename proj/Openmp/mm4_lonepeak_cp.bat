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
./sym | tee -a lonepeak_sym.$SLURM_JOB_ID\.log
