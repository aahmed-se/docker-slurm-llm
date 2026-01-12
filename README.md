# Distributed Training with PyTorch and Slurm

A minimal, educational example demonstrating distributed deep learning using PyTorch and Slurm in a Docker-based multi-node cluster.

## Overview

This project showcases **Tensor Parallelism** across multiple compute nodes using:

- **PyTorch Distributed** for multi-process training
- **Slurm** for workload management and task scheduling
- **Docker Compose** for simulating a multi-node cluster locally

This is meant for learning distributed training concepts without access to a real HPC cluster.

## Features

- **2-Node Slurm Cluster** running in Docker containers
- **Tensor Parallel Training** with model partitioning across nodes
- **Automatic Node Distribution** using Slurm's resource constraints
- **Synchronized Training** with gradient synchronization via `all_reduce`
- **Character-level Language Model** trained on Shakespeare text

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Slurm Controller                       │
│                         (node-1)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼──────┐            ┌───────▼──────┐
        │   Worker 1   │            │   Worker 2   │
        │   (node-1)   │            │   (node-2)   │
        │   Rank 0     │◄──────────►│   Rank 1     │
        │              │  all_reduce│              │
        └──────────────┘            └──────────────┘
               ▲                           ▲
               │                           │
         Model Shard 1                Model Shard 2
         (32 features)                (32 features)
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Basic understanding of distributed training concepts

### 1. Clone and Build

```bash
git clone <your-repo-url>
cd docker-slurm-llm
docker compose build
```

### 2. Start the Cluster

```bash
docker compose up -d
```

Verify both nodes are running:
```bash
docker exec node-1 sinfo -N -l
```

Expected output:
```
NODELIST   NODES PARTITION       STATE CPUS
node-1         1    debug*        idle    1
node-2         1    debug*        idle    1
```

### 3. Prepare Training Data

```bash
docker exec node-1 python3 /data/prepare_data.py
```

This downloads the Tiny Shakespeare dataset and builds the vocabulary.

### 4. Submit Training Job

```bash
docker exec node-1 sbatch /data/train.sh
```

Monitor the job:
```bash
docker exec node-1 squeue
docker exec node-1 tail -f /data/training_job.log
```

### 5. Verify Distribution

Check the log file for evidence of multi-node execution:

```bash
docker exec node-1 grep "SLURMD_NODENAME" /data/training_job.log
```

You should see:
```
[Rank: 0 | SLURMD_NODENAME: node-1 | ...]
[Rank: 1 | SLURMD_NODENAME: node-2 | ...]
```

### 6. Cleanup

```bash
docker compose down
```

## Project Structure

```
.
├── Dockerfile                 # Container definition with Slurm + PyTorch
├── docker-compose.yml         # Multi-node cluster orchestration
├── configs/
│   ├── slurm.conf            # Slurm cluster configuration
│   └── cgroup.conf           # Resource control configuration
├── scripts/
│   └── entrypoint.sh         # Container initialization script
└── workspace/
    ├── prepare_data.py       # Dataset download and preprocessing
    ├── train.sh              # Slurm job submission script
    ├── train_tp.py           # Distributed training implementation
    └── training_job.log      # Training output (generated)
```

## How It Works

### Slurm Configuration

The key to guaranteeing multi-node distribution:

```bash
# slurm.conf
NodeName=node-1 NodeAddr=node-1 CPUs=1 State=UNKNOWN
NodeName=node-2 NodeAddr=node-2 CPUs=1 State=UNKNOWN

SelectType=select/cons_tres
SelectTypeParameters=CR_Core
```

With `CPUs=1` per node and `--cpus-per-task=1`, Slurm **must** distribute tasks across both nodes.

### Training Job Script

```bash
#!/bin/bash
#SBATCH --nodes=2             # Use 2 nodes
#SBATCH --ntasks=2            # Launch 2 processes
#SBATCH --ntasks-per-node=1   # 1 process per node
#SBATCH --cpus-per-task=1     # 1 CPU per process

srun --nodelist=node-1,node-2 python3 /data/train_tp.py
```

### Tensor Parallelism Implementation

The model is partitioned across nodes:

```python
class TPLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        # Each node gets a shard of the output dimension
        self.partition_dim = output_dim // world_size
        self.weight = nn.Parameter(torch.randn(input_dim, self.partition_dim))
    
    def forward(self, x):
        return torch.matmul(x, self.weight)
```

Gradients are synchronized using `all_reduce`:

```python
reduced_loss = loss.clone().detach()
dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
reduced_loss /= world_size
```

## Key Concepts Demonstrated

### 1. **Process Group Initialization**
```python
dist.init_process_group('gloo', rank=rank, world_size=world_size)
```

### 2. **Model Sharding**
Each rank owns a partition of the model parameters (32 out of 64 output features).

### 3. **Collective Communication**
All ranks synchronize loss values using `all_reduce` for consistent training.

### 4. **SPMD (Single Program, Multiple Data)**
Same code runs on all nodes, but operates on different model shards.

## Troubleshooting

### Both processes on same node?

Check the actual distribution:
```bash
docker exec node-1 grep "SLURMD_NODENAME" /data/training_job.log
```

If both show `node-1`, ensure:
1. Slurm configuration has `CPUs=1` per node
2. Job script uses `--cpus-per-task=1`
3. Cluster was restarted after config changes

### Munge authentication errors?

The Dockerfile generates a random munge key. If nodes can't communicate:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Job stuck in queue?

Check node status:
```bash
docker exec node-1 scontrol show nodes
```

Both nodes should show `State=IDLE`.

## Scaling Up

To add more nodes, modify `docker-compose.yml` and `slurm.conf`:

```yaml
# docker-compose.yml
services:
  node-3:
    build: .
    image: slurm-llm-image
    hostname: node-3
    # ...
```

```bash
# slurm.conf
NodeName=node-3 NodeAddr=node-3 CPUs=1 State=UNKNOWN
PartitionName=debug Nodes=node-[1-3] Default=YES
```

Then update `train.sh`:
```bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
```

## Learning Resources

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Tensor Parallelism Explained](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)

## Acknowledgments

- Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) by Andrej Karpathy
- Inspired by real-world HPC training workflows