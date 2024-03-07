#!/bin/bash

#SBATCH --job-name=lieberta_spectralanalysis_code:yellow_job		  # create a short name for your job
#SBATCH --error=error
#SBATCH --nodes=1                         # node count

#SBATCH --ntasks=1                        # total number of tasks across all nodes

#SBATCH --partition=small_gpu

#SBATCH --time=23:59:00                   # total run time limit (HH:MM:SS)

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lieberta@hsu-hh.de

#SBATCH --output=terminal_output.log
 
module purge

module load anaconda3/2021.11

module load cuda

eval "$(conda shell.bash hook)"

source ~/.virtualenvs/geospatial-time-series/bin/activate

conda info --envs
 
python main.py
