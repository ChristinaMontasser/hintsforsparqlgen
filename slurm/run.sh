#!/bin/bash
#SBATCH --job-name=pretrained_model
#SBATCH --output=logs/pretrained_model_%j.out
#SBATCH --error=logs/pretrained_model_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=romeo
#SBATCH --time=10:00:00
#SBATCH --mem=64G

module spider Anaconda3/2023.09-0
module load Anaconda3

eval "$(conda shell.bash hook)"

# Now activate your environment
conda activate ENV # replace

# export HF_HOME=$HOME/.cache/huggingface
# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_HUB_CACHE=$HF_HOME/hub

export WS=export HF_HOME=$WS/models
echo "Working directory set to $WS"
echo "HF_HOME set to $HF_HOME"
# Optional: check Python
which python
python --version


echo "Running"
python main.py +experiment=c0


# Wait for the process to finish
wait


