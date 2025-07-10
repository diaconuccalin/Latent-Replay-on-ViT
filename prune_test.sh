#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=00:30:00              # time limit: 1 hour
#SBATCH --error=test_prune.err       # standard error file
#SBATCH --output=test_prune.out      # standard output file

python main.py --pipeline "block_level_pruning" --session_name "test_pruning"
