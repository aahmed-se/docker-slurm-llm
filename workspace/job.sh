#!/bin/bash
#SBATCH --job-name=llm-tp
#SBATCH --output=/data/training_job.log

# --- ALLOCATION SETTINGS ---
#SBATCH --nodes=2             # Allocate 2 distinct nodes
#SBATCH --ntasks=2            # Allocate 2 tasks total
#SBATCH --ntasks-per-node=1   # Limit to 1 task per node
#SBATCH --cpus-per-task=1     # Request 1 CPU per task (GUARANTEES multi-node)
# ---------------------------

# Verify data exists
if [ ! -f "/data/data/meta.json" ]; then
    echo "Error: Data not found. Please run prepare_data.py first."
    exit 1
fi

echo "------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Master Node: $(hostname)"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Strategy: --cpus-per-task=1 with CPUs=1 per node forces 2-node distribution"
echo "------------------------------------------------------"

# --- LAUNCH COMMAND ---
# Explicitly specify node list to force distribution
# This ensures Task 0 -> node-1, Task 1 -> node-2
srun --nodelist=node-1,node-2 python3 /data/train_tp.py