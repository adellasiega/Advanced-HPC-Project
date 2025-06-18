#!/bin/bash

#SBATCH --job-name=compile_jacobi
#SBATCH -A ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem=1000
#SBATCH --time=00:01:00

# set -x

# Load the required modules
module load nvhpc/24.3 openmpi/4.1.6--nvhpc--24.3

# Build the project
make

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful :)"
else
    echo "Compilation failed :("
    exit 1
fi

