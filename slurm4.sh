#!/bin/bash
#SBATCH -J mech_2n256
#SBATCH -p fankomp
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH -o x2n256_impi.out

module purge
module load ansys/2025R1
module load oneapi/2025.2            # provides Intel MPI
# oneAPI usually auto-sources setvars; if not:
# source /opt/intel/oneapi/setvars.sh

export OMP_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=verbs;ofi_rxm
export FI_PROVIDER=verbs
export I_MPI_DEBUG=5

export ANSYS_SCR=/tmp/${SLURM_JOBID:-$$}
srun --ntasks-per-node=1 bash -lc 'mkdir -p '"$ANSYS_SCR"

# Option 1: use mpirun from Intel MPI
mpirun -bootstrap slurm -np $SLURM_NTASKS \
  ansys232 -b -dis -p mech -i ds.dat -o file.out -np $SLURM_NTASKS -dir $ANSYS_SCR

rm -f file[0-9]*
