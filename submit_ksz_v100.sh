#!/bin/bash
#SBATCH --job-name=kSZ_v100
#SBATCH --partition=crmda
#SBATCH --ntasks=1            # 1 task
#SBATCH --constraint=ib
#SBATCH --time=0-06:00:00       # Time limit days-hrs:min:sec
#SBATCH --gres=gpu            # For v100 --gres=gpu --constraint=v100
#SBATCH --mail-user=nesar@ku.edu
#SBATCH --mail-type=ALL
#SBATCH -D /home/n335r736/kSZ/kSZ
#SBATCH -e V100_%x-%A.err
#SBATCH -o V100_%x-%A.out


 
source activate condampi4py
module load singularity
CONTAINERS=/panfs/pfs.local/software/install/singularity/containers
singularity exec --nv $CONTAINERS/tensorflow-gpu-1.9.0.img python ksz_Zsnap_multi_linear.py

#mpirun python kaiser_cor_cf3_3D_OuterRim_psi_cluster_LG.py
#python ksz_Zsnap_multi_linear.py 
