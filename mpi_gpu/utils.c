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

void write_heatmap(double* M, size_t nx, size_t ny) {
    double v_max = 0.0;
    for (size_t i = 1; i < nx; ++i)
        for (size_t j = 1; j < ny; ++j)
            if (M[i * ny + j] > v_max)
                v_max = M[i * ny + j];

    FILE* out = fopen("heatmap.ppm", "wb");
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

void write_results(size_t nx, size_t ny, size_t n_iterations, int size, int num_threads, double total_time, double init_time, double comm_time, double comp_time) {        
    FILE* log_file = fopen("timing_results.txt", "a");
    if (log_file == NULL) {
        fprintf(stderr, "Failed to open timing_results.txt for writing\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(log_file, "Run with nx=%zu, ny=%zu, it=%zu, MPI_size=%d, OMP_num_threads=%d\n", nx, ny, n_iterations, size, num_threads);
    fprintf(log_file, "Total time elapsed: %.6f s\n", total_time);
    fprintf(log_file, "Total initialization time: %.6f s\n", init_time);
    fprintf(log_file, "Total communication time: %.6f s\n", comm_time);
    fprintf(log_file, "Total computing time: %.6f s\n\n", comp_time);
    fclose(log_file);
}
