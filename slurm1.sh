#!/bin/bash
#SBATCH -J mech_2n256
#SBATCH -p fankomp                 # change if needed
#SBATCH -N 2
#SBATCH --ntasks-per-node=128      # 128 cores per node → 256 ranks total
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH -o x2n256.out

module purge
module load ansys/2023R2           # or your version (e.g., ansys/2024R1)
module load openmpi ucx            # OpenMPI built with UCX

# prevent TCP fallback; use IB/RoCE only
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^tcp
export OMPI_MCA_oob=^tcp
export UCX_TLS=rc,sm,self          # RoCE: rc_mlx5,sm,self
# pick your HCA:port; if unsure, leave unset and check ucx_info -d
export UCX_NET_DEVICES=mlx5_0:1
export UCX_LOG_LEVEL=warn

# fast local scratch on each node
export ANSYS_SCR=/tmp/${SLURM_JOBID:-$$}
srun --ntasks-per-node=1 bash -lc 'mkdir -p '"$ANSYS_SCR"

# run Mechanical (distributed, all-MPI, 256 ranks, no OpenMP)
srun --mpi=pmix ansys232 \
  -b -dis -p mech \
  -i ds.dat -o file.out \
  -np $SLURM_NTASKS \
  -dir $ANSYS_SCR

# cleanup
rm -f file[0-9]*
