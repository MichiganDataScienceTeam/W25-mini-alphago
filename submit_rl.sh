#!/bin/bash

# Reinforcement learning (self-play) from a randomly initialized model
# Saves RL_weights.pt and RL_ds.pt after every epoch
# and runs until the time limit is reached (EPOCHS in config.py is very large)
# Usage: sbatch submit_rl.sh

# SLURM directives

#SBATCH --account=mdatascience_team  # Account to charge the job to
#SBATCH --job-name=alphago-rl       # Job name
#SBATCH --mail-user=lujeff@umich.edu  # Email for notifications
#SBATCH --mail-type=END,FAIL        # Notification type: send email when job ends or fails
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=33          # Number of CPUs per task (PROCESSES in config.py + 1)
#SBATCH --mem=64G                   # Total memory for the job (~0.5GB per self-play worker)
#SBATCH --time=1-00:00:00           # Maximum runtime (1 day max for the team QOS)
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --partition=gpu             # Partition to use
#SBATCH --output=%x-%j.log          # Log file (job name and id)


# Environment setup

module purge                        # Clean environment
module load python3.11-anaconda/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alphago || { echo "Failed to activate conda environment"; exit 1; }

cd "$SLURM_SUBMIT_DIR"

# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Self-play requires an empty temp directory (may be left over from a killed job)
rm -rf temp_files

# Log GPU use every 5 minutes (low use during self-play means the CPUs are the bottleneck)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 300 > "gpu-$SLURM_JOB_ID.csv" &


# Run the program
python -u mp_train_alphago.py
