#!/bin/bash
#SBATCH --job-name=jacobi
#SBATCH -A ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=100GB
#SBATCH --time=00:05:00

# Load the required modules
module load nvhpc/24.3 openmpi/4.1.6--nvhpc--24.3

# Set the number of tasks and threads
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=8

# Set the OMP environment variables
export OMP_PLACES=cores 
export OMP_PROC_BIND=true

# Set the size of the matrix and the number of iterations
export N=10000
export ITERS=1000

# Directories for results
cd $SLURM_SUBMIT_DIR
mkdir -p one-side-communication/results
mkdir -p two-side-communication/results

# Run the two versions of th algorithm with the specified parameters
srun --cpu-bind=verbose ./one-side-communication/executable $N $N $ITERS
srun --cpu-bind=verbose ./two-side-communication/executable $N $N $ITERS

