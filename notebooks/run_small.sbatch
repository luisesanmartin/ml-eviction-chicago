#!/bin/bash
#SBATCH --job-name=ML_analysis_sbatch
#SBATCH --output=ML_analysis_sbatch.out
#SBATCH --error=ML_analysis_sbatch.err
#SBATCH --time=05:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load python/3.7.0
python3 ML-analysis_small.py