#!/bin/bash

#SBATCH --job-name=jobt.job
#SBATCH --output=jobt.out
#SBATCH --export=ALL

#SBATCH --partition=dev_gpu_4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3
#SBATCH --mem-per-gpu=94000mb
#SBATCH --time=00:30:00

#SBATCH --mail-user=max.mohr@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate conda environment
source /home/tu/tu_tu/tu_zxonr37/miniconda3/etc/profile.d/conda.sh
conda activate tue_mthesis_linux
# Check if conda environment is activated
conda info --envs

# Disable numa balancing
# echo 0 > /proc/sys/kernel/numa_balancing # Permission denied

# Start ollama in the background
CUDA_VISIBLE_DEVICES=0,1,2 OLLAMA_DEBUG=1 ollama serve &
ollama list
ollama ps
# Wait a few seconds to ensure ollama has started
sleep 5

# Execute python script
# python -m src.main -c=True -m="llama3.1:70b"
python -m src.cluster.upload_data