#!/bin/bash

#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --time=20:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task (adjust as needed)
#SBATCH --partition=boost_usr_prod            # Options: lrd_all_serial, boost_usr_prod
#SBATCH --qos=normal                          # Options: normal, boost_qos_dbg (debugging), boost_qos_bprod (big prod), boost_qos_lprod (light prod)
#SBATCH --gres=gpu:1                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --output=gpu_prune.out                # File for standard output
#SBATCH --error=gpu_prune.err                 # File for standard error
#SBATCH --account=iscrc_demollm

# Load necessary modules (adjust to your environment)
module load cuda/12.2                         # Load CUDA toolkit

# Optional: Set environment variables for performance tuning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # Set OpenMP threads per task
export NCCL_DEBUG=INFO                        # Enable NCCL debugging (for multi-GPU communication)

# Launch the distributed GPU application
# Replace with your actual command (e.g., mpirun or srun)
# srun --mpi=pmix ./my_distributed_gpu_app --config config.yaml
python main.py --pipeline "block_level_pruning" --session_name "vit_b_16_default"
