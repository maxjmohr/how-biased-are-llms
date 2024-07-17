#!/bin/bash

#SBATCH --job-name=job.job
#SBATCH --output=job.out
#SBATCH --export=ALL

#SBATCH --partition=dev_gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100000mb
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

#SBATCH --mail-user=max.mohr@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate conda environment
source activate tue_mthesis_linux

# Start ollama in the background
ollama serve &
# Wait a few seconds to ensure ollama has started
sleep 5

# Execute python script
python master_thesis/03_Codebase/src/models.py