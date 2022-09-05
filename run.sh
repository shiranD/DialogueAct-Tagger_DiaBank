#!/bin/bash
#!/bin/bash

### --------  SLURM  ----------- ###
#SBATCH --job-name=training
#SBATCH --qos=blanca-kann
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="out/train_%A_%a_%j.out"
#SBATCH --error="error/train_%A_%a_%j.err"
#SBATCH --mem=64G
### -------------------------- ###

module load cuda/11.2
export LD_LIBRARY_PATH=/curc/sw/cudnn/8.1_for_cuda_11.2/cuda/lib64:/curc/sw/cuda/11.2/lib64

source /projects/shdu9019//miniconda3/etc/profile.d/conda.sh
conda init
conda activate DA_36

python scripts/train_transformer.py
