#!/bin/bash
#SBATCH --job-name=SRBUILD
#SBATCH --output=/home/mila/r/rebecca.salganik/Projects/MusicSAGE/logs/SLURM/R_%j_output.txt
#SBATCH --error=/home/mila/r/rebecca.salganik/Projects/MusicSAGE/logs/SLURM/R_%j_error.txt 
#SBATCH --mail-user=rebecca.salganik@gmail.com  
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --mem=100Gb
#SBATCH --array=0-0:1
#SBATCH -c 4

module load anaconda/3
conda activate nsv5

dt=$(date '+%d/%m/%Y %H:%M:%S');

echo "$dt" + 'running experiment' + $1 + 'jobid' + $SLURM_JOB_ID + 'task id' + $SLURM_ARRAY_TASK_ID >> /home/mila/r/rebecca.salganik/Projects/MusicSAGE/logs/All_Experiments.txt

python -u /home/mila/r/rebecca.salganik/Projects/MusicSAGE/src2/benchmarks/ScoreReg/pre_build.py

##/home/mila/r/rebecca.salganik/Projects/MusicSAGE/src2/benchmarks/ScoreReg/runner.py