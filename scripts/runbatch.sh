#!/bin/bash
 
# Request resources (per task):
#SBATCH -c 1           # 1 CPU core
#SBATCH --mem=16G       # 1 GB RAM
#SBATCH --time=12:0:0   # 6 hours (hours:minutes:seconds)
 
# Run on the shared queue
#SBATCH -p shared
 
# Specify the tasks to run:
#SBATCH --array=1-8   # Number of tasks to be run

#SBATCH --output=./batch_logs/output_%a_%A.txt
 
# Each separate task can be identified based on the SLURM_ARRAY_TASK_ID
# environment variable:
 
echo "I am task number $SLURM_ARRAY_TASK_ID"
 
# Run program:
module load python
source ../.venv/bin/activate
python 0_runbatch_time.py 4
