#!/bin/bash
# Independent original-seed replay; override --exclude for DT seed0 / v3 seed2.
#SBATCH --job-name=v3-replay012
#SBATCH --partition=GPUNorm
#SBATCH --gres=gpu:1
#SBATCH --exclude=gn1,gn2,gn3,gn6,gn7,gn8,gn9,gn10,gn11,gn12,gn13
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/labmount/users/202615385/code/DT-EXP/logs/v3-replay012-%j.out
set -euo pipefail
PROJECT=/labmount/users/202615385/code/DT-EXP
exec bash "$PROJECT/scripts/dt_experiments/run_v3_high_legacy345.sbatch" "$@" --profile replay012
