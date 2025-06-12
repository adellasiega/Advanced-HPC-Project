#!/bin/bash
#SBATCH --job-name=jacobi
#ACCOUNT --account=DSSC
#SBATCH --partition=GPU
#SBATCH --mem=8gb
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=jacobi_%j.out

module load openMPI

mpicc -O3 -march=native -Wall -Wextra -Wpedantic -Werror -lm -fast -Minfo=all -v -Mneginfo -gpu=cc80 -target=gpu -gpu=nomanaged -mp=multicore,gpu jacobi.c -o jacobi.x

# Set the number of tasks and threads
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=8
# Set the OMP environment variables
export OMP_PLACES=cores 
export OMP_PROC_BIND=true

# Set the size of the matrix and the number of iterations
export N=10000
export ITERS=1000

# Run the program with the specified parameters
srun --cpu-bind=verbose ./jacobi.x $N $N $ITERS
