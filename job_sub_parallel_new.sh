#!/bin/bash
#SBATCH --job-name=kSZ-CPU
#SBATCH --partition=crmda
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=10
#SBATCH --constraint=ib
#SBATCH --mem=100gb
#SBATCH --time=48:00:00
#SBATCH --mail-user=n335r736@ku.edu
#SBATCH --mail-type=ALL
#SBATCH -D /home/n335r736/kSZ
#SBATCH -e /home/n335r736/kSZ/kSZ/%x-%A.err
#SBATCH -o /home/n335r736/kSz/kSZ/%x-%A.out

####source activate condampi4py_new
source activate /panfs/pfs.local/work/physastro/n335r736/ml_env
mpirun python ksz_Zsnap_multi_linear_Cluster.py
