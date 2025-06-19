#ifndef UTILS_H
#define UTILS_H

double* alloc_init_matrix(size_t nx, size_t ny); 
void write_heatmap(const char* filename, double* M, size_t nx, size_t ny);
void write_results(const char* filename, size_t nx, size_t ny, size_t n_iterations, int n_nodes, int size, int num_threads, double total_time, double init_time, double comm_time, double comp_time);
int get_num_nodes_from_env();

#endif
