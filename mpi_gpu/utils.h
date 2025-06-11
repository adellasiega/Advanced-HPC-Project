#ifndef UTILS_H
#define UTILS_H

double* alloc_init_matrix(size_t nx, size_t ny); 
void write_heatmap(double* M, size_t nx, size_t ny);
void write_results(size_t nx, size_t ny, size_t n_iterations, int size, int num_threads, double total_time, double init_time, double comm_time, double comp_time);

#endif
