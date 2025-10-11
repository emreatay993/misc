#!/bin/bash
#SBATCH -J mech_2n256
#SBATCH -p fankomp
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH -o x2n256.out

module purge
module load ansys/2025R1
module load ucx/1.15.0 pmix/4.2.9 openmpi/4.1.8

# UCX/OMPI: prevent TCP fallback
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^tcp
export OMPI_MCA_oob=^tcp
export UCX_TLS=rc,sm,self             # RoCE: rc_mlx5,sm,self
export UCX_NET_DEVICES=mlx5_0:1       # set to your HCA:port
export UCX_LOG_LEVEL=warn

# node-local scratch
export ANSYS_SCR=/tmp/${SLURM_JOBID:-$$}
srun --ntasks-per-node=1 bash -lc 'mkdir -p '"$ANSYS_SCR"

# run Ansys distributed, all-MPI (256 ranks)
srun --mpi=pmix ansys232 -b -dis -p mech -i ds.dat -o file.out -np $SLURM_NTASKS -dir $ANSYS_SCR

rm -f file[0-9]*
