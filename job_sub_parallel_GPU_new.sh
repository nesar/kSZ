#!/bin/bash
#SBATCH --job-name=kSZ-GPU
#SBATCH --partition=sixhour
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=10
#SBATCh --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mail-user=n335r736@ku.edu
#SBATCH --mail-type=ALL
#SBATCH -D /home/n335r736/kSZ
#SBATCH -e /home/n335r736/report/%x-%A.err
#SBATCH -o /home/n335r736/report/%x-%A.out

######source activate condampi4py
source activate /panfs/pfs.local/work/physastro/n335r736/ml_env

python ksz_Zsnap_multi_linear_Cluster.py 
