#!/bin/bash
  
# SLURM directives

#SBATCH --account=mdatascience_team  # Account to charge the job to
#SBATCH --job-name=alpha-go    # Job name
#SBATCH --mail-user=ozeronat@umich.edu  # Email for notifications
#SBATCH --mail-type=END             # Notification type: send email when job ends
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=2           # Number of CPUs per task
#SBATCH --mem=184320M                   # Total memory for the job
#SBATCH --time=14:00:00             # Maximum runtime
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --partition=gpu             # Partition to use


# Environment setup

module purge                         # Clean environment
module load python/3.9 cuda/11.8     # Load necessary modules (update versions as required)

# Check GPU and CUDA availability
echo "CUDA and GPU Details:"

nvidia-smi
nvcc --version


# Activate virtual environment
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }


# Ensure required packages are installed
pip install -r requirements.txt
chmod -x install_libs.sh
./istall_libs.sh

# Run the program
python3 train_alphago.py

# Post-job cleanup (optional)
deactivate