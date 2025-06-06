#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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

void heatmap(double* M, size_t nx, size_t ny) {
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
    printf("Heatmap saved to heatmap.ppm\n");
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s nx ny n_iterations\n", argv[0]);
        return 1;
    }
    
    /// Get dimensions and number of iterations
    size_t nx = (size_t)atoi(argv[1]);
    size_t ny = (size_t)atoi(argv[2]);
    size_t n_iterations = (size_t)atoi(argv[3]);

    /// Timing variables
    double total_start, total_end;
    double init_start, init_end;
    double comm_start, comm_end, comm_time = 0.0;
    double comp_start, comp_end, comp_time = 0.0;

    /// Set up multithread MPI
    int required = MPI_THREAD_FUNNELED;
    int provided; 
    MPI_Init_thread(&argc, &argv, required ,&provided);

    if (provided != required) {
        printf("MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /// Get the number of omp threads
    #pragma omp parallel
    {
        #pragma omp single
        printf("Running with %d threads on rank %d\n", omp_get_num_threads(), rank);
    }

    /// Start total time
    MPI_Barrier(MPI_COMM_WORLD); 
    total_start = MPI_Wtime();
    
    /// Start initialization time
    MPI_Barrier(MPI_COMM_WORLD); 
    init_start = MPI_Wtime();

    /// Number of rows to assing to each process
    size_t local_nx_no_bound = nx / size + (rank < (nx % size) ? 1 : 0);
    size_t local_nx = local_nx_no_bound;
    if (size > 1) {
        if (rank == 0 || rank == size - 1)
            local_nx += 1;
        else
            local_nx += 2;
    }
    
    /// Allocate sub matrixes for calculation
    double* M = (double*)calloc(local_nx * ny, sizeof(double));
    double* M_new = (double*)calloc(local_nx * ny, sizeof(double));
    
    /// Complete matrix
    double* M_full = NULL;

    /// Scatterv info
    int* sendcounts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        M_full = alloc_init_matrix(nx, ny);
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int rows = r < (nx % size) ? nx / size + 1 : nx / size;
            sendcounts[r] = rows * ny;
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    /// Scatter the complete matrix among processes
    MPI_Scatterv(rank == 0 ? M_full : NULL, sendcounts, displs, MPI_DOUBLE,
                 rank == 0 ? M : M + ny, local_nx_no_bound * ny, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    /// Define north and south process w.r.t. current process 
    /// to comunicate boundary rows of the submatrixes
    int proc_north = (rank - 1 >= 0) ? rank - 1 : MPI_PROC_NULL;
    int proc_south = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    
    /// Stop initialization time
    MPI_Barrier(MPI_COMM_WORLD); 
    init_end = MPI_Wtime();
    double init_time = init_end - init_start;

    /// Jacobi algorithm
    for (size_t iteration = 0; iteration < n_iterations; ++iteration) {
        
        /// Start Communication Time
        comm_start = MPI_Wtime();
        
        /// Send and receive boundary rows
        MPI_Sendrecv(M + ny, ny, MPI_DOUBLE, proc_north, 0,
                     M + (local_nx - 1) * ny, ny, MPI_DOUBLE, proc_south, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(M + (local_nx - 2) * ny, ny, MPI_DOUBLE, proc_south, 1,
                     M, ny, MPI_DOUBLE, proc_north, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        /// Partial communication time
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
        
        /// Computation time
        comp_start = MPI_Wtime();

        /// Jacobi iteration
        #pragma omp parallel for collapse(2)
        for (size_t i = 1; i < local_nx - 1; ++i) {
            for (size_t j = 1; j < ny - 1; ++j) {
                M_new[i * ny + j] = 0.25 * (M[(i - 1) * ny + j] + M[i * ny + j + 1] +
                                           M[(i + 1) * ny + j] + M[i * ny + j - 1]);
            }
        }
        
        for (size_t j = 0; j < ny; ++j) {
            M_new[0 * ny + j] = M[0 * ny + j];
            M_new[(local_nx - 1) * ny + j] = M[(local_nx - 1) * ny + j];
        }

        for (size_t i = 0; i < local_nx; ++i) {
            M_new[i * ny + 0] = M[i * ny + 0];
            M_new[i * ny + (ny - 1)] = M[i * ny + (ny - 1)];
        }

        /// Partial computation time
        comp_end = MPI_Wtime();
        comp_time += comp_end - comp_start;
        
        /// Swap M and M_new for next iteration
        double* tmp = M;
        M = M_new;
        M_new = tmp;
    }

    /// Collect submatrixes in full matrix
    MPI_Barrier(MPI_COMM_WORLD);
    comm_start = MPI_Wtime();

    MPI_Gatherv(rank == 0 ? M : M + ny, local_nx_no_bound * ny, MPI_DOUBLE,
                M_full, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;


    MPI_Barrier(MPI_COMM_WORLD);
    total_end = MPI_Wtime();
    double total_time = total_end - total_start;
    
    if (rank == 0) {
        heatmap(M_full, nx, ny);
    }
    
    free(M);
    free(M_new);
    if (rank == 0) {
        free(M_full);
        free(sendcounts);
        free(displs);
    }
    
    if (rank == 0) {
        printf("Total time elapsed: %.6f s\n", total_time);
        printf("Total initialization time: %.6f s\n", init_time);
        printf("Total communication time: %.6f s\n", comm_time );
        printf("Total computing time: %.6f s\n", comp_time);
    };
    
    MPI_Finalize();
    return 0;
}

