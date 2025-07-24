#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>
#include "../utils.h"

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

    /// Set multithread MPI
    int required = MPI_THREAD_FUNNELED;
    int provided; 
    MPI_Init_thread(&argc, &argv, required ,&provided);

    if (provided != required) {
        printf("MPI does not provide required threading level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /// Get rank and number of MPI processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /// Get the number of OMP threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    /// Get the number of devices
    const int num_gpus = omp_get_num_devices();
    const int device_id = rank % num_gpus;
    omp_set_default_device(device_id);

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
    /// to comunicate submatrixes' boundary rows
    int proc_north = (rank - 1 >= 0) ? rank - 1 : MPI_PROC_NULL;
    int proc_south = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;

    /// Stop initialization time
    MPI_Barrier(MPI_COMM_WORLD); 
    init_end = MPI_Wtime();
    double init_time = init_end - init_start;
    
    /// Define RMA windows
    MPI_Win win_north;
    MPI_Win win_south;
    
    int total_size=local_nx*ny;
    #pragma omp target data map(tofrom: M[0:total_size]) map(to: M_new[0:total_size])
    {
	    #pragma omp target data use_device_ptr(M)
	    {
		    MPI_Win_create(
			    M + ny,
			    ny * sizeof(double),
			    sizeof(double),
			    MPI_INFO_NULL,
			    MPI_COMM_WORLD,
			    &win_north);
		    
		    MPI_Win_create(
			    M + (local_nx - 2) * ny, 
			    ny * sizeof(double),
			    sizeof(double),
			    MPI_INFO_NULL, 
			    MPI_COMM_WORLD, 
			    &win_south);
	    } 
	    
	    /// Jacobi algorithm
	    for (size_t iteration = 0; iteration < n_iterations; ++iteration) {
            

            /// Start communication time
            comm_start = MPI_Wtime();
           
            /// Communicate boundary rows
            #pragma omp target data use_device_ptr(M)
            {
                MPI_Win_fence(0, win_north);
                MPI_Win_fence(0, win_south);

                MPI_Get(
                    M + (local_nx - 1) * ny, ny, MPI_DOUBLE, 
                    proc_south, 0, ny, MPI_DOUBLE, 
                    win_north
                );

                 MPI_Get(
                    M, ny, MPI_DOUBLE, 
                    proc_north, 0, ny, MPI_DOUBLE, 
                    win_south
                );

                MPI_Win_fence(0, win_north);
                MPI_Win_fence(0, win_south);
            }

            /// Stop communication time
            comm_end = MPI_Wtime();
            comm_time += comm_end - comm_start;
            
            /// Start computation time
            comp_start = MPI_Wtime();

            /// Jacobi iterations
            #pragma omp target teams distribute parallel for simd collapse(2) num_teams(108)  
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

            /// Stop computation time
            comp_end = MPI_Wtime();
            comp_time += comp_end - comp_start;
            
            /// Swap M and M_new for next iteration
            double* tmp = M;
            M = M_new;
            M_new = tmp;
        }
    }

    /// Start communication time
    MPI_Barrier(MPI_COMM_WORLD);
    comm_start = MPI_Wtime();

    /// Collect submatrixes in full matrix
    MPI_Gatherv(rank == 0 ? M : M + ny, local_nx_no_bound * ny, MPI_DOUBLE,
                M_full, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    /// Stop communication time
    MPI_Barrier(MPI_COMM_WORLD); 
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;

    /// Stop total time
    MPI_Barrier(MPI_COMM_WORLD);
    total_end = MPI_Wtime();
    double total_time = total_end - total_start;
    
    /// Write results
    if (rank == 0) {
	int n_nodes = get_num_nodes_from_env();
        write_heatmap("./one-side-communication/results/heatmap.ppm", M_full, nx, ny);
        write_results("./one-side-communication/results/timing.txt", nx, ny, n_iterations, n_nodes, size, num_threads, total_time, init_time, comm_time, comp_time);
    }
    
    ///Deallocate resources 
    MPI_Win_free(&win_north);
    MPI_Win_free(&win_south);
    free(M);
    free(M_new);
    if (rank == 0) {
        free(M_full);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}

