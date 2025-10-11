module purge
module load ansys/2023R2 intel-mpi

export OMP_NUM_THREADS=1
export I_MPI_DEBUG=5
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=verbs;ofi_rxm   # RoCE: rxm
export FI_PROVIDER=verbs                  # prevent TCP

srun --mpi=pmix ansys232 \
  -b -dis -p mech \
  -i ds.dat -o file.out \
  -np $SLURM_NTASKS \
  -dir $ANSYS_SCR
