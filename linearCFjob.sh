#!/bin/bash
#SBATCH --job-name=linearCF
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mem=10G

module purge
module restore bao
conda activate bao

python project/bao_sims/computeLinearCF.py

