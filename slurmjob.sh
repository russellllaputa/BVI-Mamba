#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=coms030646
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
#SBATCH --output=log-job.out
#SBATCH --error=log-err.err
#SBATCH --nodelist=bp1-gpu030



conda activate /user/home/kc24142/miniconda3/envs/Retinexformer

cd /user/work/kc24142/BVI-Mamba

python train.py --config STASUNet-DID.yml
