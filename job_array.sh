#!/bin/bash
#SBATCH --job-name=pyjob
#SBATCH --output=data/logs/love_numbers/job_%A_%a.out
#SBATCH --error=data/logs/love_numbers/job_%A_%a.err
#SBATCH --array=0-0  # Dummy default, overridden by Python via sbatch

FUNCTION=$1
JOB_ARRAY_NAME=$2
JOB_ARRAY_MAX_FILE_SIZE=$3
JOB_ID=$SLURM_ARRAY_TASK_ID

python "worker_${FUNCTION}.py" "$JOB_ARRAY_NAME" "$JOB_ARRAY_MAX_FILE_SIZE" "$JOB_ID"
