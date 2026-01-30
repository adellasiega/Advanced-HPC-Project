# Parallel Heat Equation Solver

A parallel implementation of the 2D heat equation using the Jacobi iterative method with MPI for distributed computing.

## Overview

This program solves the 2D steady-state heat equation using domain decomposition across multiple MPI processes. Each process handles a subset of rows, communicating boundary data with neighbors to iteratively converge to the solution.

## Features

- **MPI Parallelization**: Domain decomposition with row-wise distribution
- **Jacobi Iteration**: Standard iterative solver for the heat equation
- **Visualization**: Generates PPM heatmap of the final temperature distribution
- **Load Balancing**: Automatically distributes remainder rows when domain size isn't evenly divisible

## Requirements

- C++11 or later
- MPI implementation (OpenMPI, MPICH, etc.)
- C++ compiler with MPI support (mpic++, mpicxx)

## Compilation
```bash
mpic++ -std=c++11 -o heat_solver heat_solver.cpp
```

## Usage
```bash
mpirun -np <num_processes> ./heat_solver <nx> <ny> <iterations>
```

**Parameters:**
- `num_processes`: Number of MPI processes
- `nx`: Number of grid points in x-direction (rows)
- `ny`: Number of grid points in y-direction (columns)
- `iterations`: Number of Jacobi iterations

**Example:**
```bash
mpirun -np 4 ./heat_solver 500 500 1000
```

## Output

The program generates `heatmap.ppm`, a PPM image file showing the temperature distribution with a color gradient:
- Red: High temperature
- Green: Medium temperature
- Blue: Low temperature

View the output with any PPM-compatible image viewer:
```bash
# Convert to PNG (requires ImageMagick)
convert heatmap.ppm heatmap.png

# Or view directly
eog heatmap.ppm  # Linux
open heatmap.ppm # macOS
```

## Algorithm

1. **Initialization**: Process 0 creates the full grid with boundary conditions
2. **Domain Decomposition**: Grid rows distributed across MPI processes
3. **Jacobi Iteration**: 
   - Exchange boundary rows between neighboring processes
   - Update interior points: `M_new(i,j) = (M(i-1,j) + M(i+1,j) + M(i,j-1) + M(i,j+1)) / 4`
   - Preserve boundary conditions
4. **Gathering**: Process 0 collects results and generates heatmap

## Boundary Conditions

- **Top edge** (i=0): Temperature = 0
- **Right edge** (j=ny-1): Temperature = 0
- **Bottom edge** (i=nx-1): Linear gradient from 100 to 0
- **Left edge** (j=0): Linear gradient from 100 to 0
- **Interior**: Initial value = 0.5

## Performance Considerations

- Best performance with `nx` divisible by number of processes
- Communication overhead increases with more processes
- Larger grids benefit more from parallelization
- Each process requires memory for local subdomain plus ghost rows

## Implementation Details

- **Matrix Class**: Template-based 2D matrix with row-major storage
- **Ghost Rows**: Processes maintain boundary rows from neighbors
- **MPI Communication**: `Sendrecv` for boundary exchange, `Scatterv`/`Gatherv` for distribution
