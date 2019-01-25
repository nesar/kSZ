#!/bin/bash
#SBATCH --job-name=kSZ_horovod
#SBATCH --partition=crmda
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --constraint=ib
#SBATCH --mem=10gb
#SBATCH --time=08:00:00
#SBATCH --mail-user=nesar@ku.edu
#SBATCH --mail-type=ALL
#SBATCH -D /home/n335r736/kSZ/kSZ
#SBATCH -e horovod_%x-%A.err
#SBATCH -o horovod_%x-%A.out

source activate /panfs/pfs.local/work/physastro/n335r736/ml_env
mpirun -np 4 python ksz_Zsnap_multi_linear_horovod.py
