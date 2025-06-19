#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

double* alloc_init_matrix(size_t nx, size_t ny) {
    double* M = (double*)malloc(nx * ny * sizeof(double));
    double v_max = 100.0;
    double v_inner = 0.5;

    // Initialize all to inner value
    for (size_t i = 0; i < nx; ++i)
        for (size_t j = 0; j < ny; ++j)
            M[i * ny + j] = v_inner;

    // Set boundary conditions
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            if (i == nx - 1) {
                double q = (double)j / (ny - 1);
                M[i * ny + j] = v_max * (1.0 - q);
            } else if (j == 0) {
                double q = (double)i / (nx - 1);
                M[(nx - i - 1) * ny + j] = v_max * (1.0 - q);
            } else if (i == 0 || j == ny - 1) {
                M[i * ny + j] = 0.0;
            }
        }
    }

    return M;
}

void write_heatmap(const char* filename, double* M, size_t nx, size_t ny) {
    double v_max = 0.0;
    for (size_t i = 1; i < nx; ++i)
        for (size_t j = 1; j < ny; ++j)
            if (M[i * ny + j] > v_max)
                v_max = M[i * ny + j];

    FILE* out = fopen(filename, "wb");
    if (out == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    fprintf(out, "P6\n%zu %zu\n255\n", ny - 2, nx - 2);

    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            double v = M[i * ny + j] / v_max;
            uint8_t r = (uint8_t)(v * 255);
            uint8_t g = (uint8_t)((1 - fabs(v - 0.5) * 2) * 255);
            uint8_t b = (uint8_t)(255 - v * 255);
            fputc(r, out);
            fputc(g, out);
            fputc(b, out);
        }
    }

    fclose(out);
}

void write_results(const char* filename, size_t nx, size_t ny, size_t n_iterations,
                       int n_nodes, int size, int num_threads, double total_time,
                       double init_time, double comm_time, double comp_time) {
    int write_header = 0;
    FILE* file_check = fopen(filename, "r");
    if (file_check == NULL) {
        write_header = 1; // File does not exist, write header
    } else {
        fclose(file_check);
    }

    FILE* log_file = fopen(filename, "a");
    if (log_file == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (write_header) {
        fprintf(log_file, "nx,ny,it,num_nodes,MPI_size,OMP_num_threads,total_time,init_time,comm_time,comp_time\n");
    }

    fprintf(log_file, "%zu,%zu,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
            nx, ny, n_iterations, n_nodes, size, num_threads,
            total_time, init_time, comm_time, comp_time);

    fclose(log_file);
}


int get_num_nodes_from_env() {
    const char* env = getenv("SLURM_NNODES");
    if (env) {
        return atoi(env);
    } else {
        fprintf(stderr, "SLURM_NNODES not set; are you running under SLURM?\n");
        return -1;
    }
}
