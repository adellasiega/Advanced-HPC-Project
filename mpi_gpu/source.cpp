#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <mpi.h>

template <typename T>
class Matrix {

    private:
    std::vector<T> data;
    size_t nx, ny; //number of rows, number of columns

    public:
    Matrix(size_t nx, size_t ny, T init_val = T())
        : data(nx * ny, init_val), nx(nx), ny(ny) {}
    
    T& operator()(size_t i, size_t j) {
        return data[i * ny + j];
    }
    const T& operator()(size_t i, size_t j) const {
        return data[i * ny + j];
    }

    T* raw() { 
        return data.data();
    }
    const T* raw() const { 
        return data.data();
    }
    
    size_t rows() const {
        return nx;
    }
    size_t cols() const {
        return ny;
    }
};

Matrix<double> alloc_init_matrix(size_t nx, size_t ny) {
    // Initial conditions
    double v_max = 100.0;
    double v_inner = 0.5;
    
    // Allocate memory and initialize matrix M
    Matrix<double> M(nx, ny, v_inner);
    
    // Set boundary conditions
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {  
            if (i == nx-1) {
                double q = static_cast<double>(j)/(ny-1);
                M(i,j) = v_max * (1.0-q); 
            } else if (j == 0) {
                double q = static_cast<double>(i)/(nx-1);
                M(nx-i-1,j) = v_max * (1.0-q);
            } else if (i == 0 || j == ny-1) {
                M(i,j) = 0;
            }
        }
    }

    return M;
}

void heatmap(Matrix<double> M) {
    // Image dimensions
    size_t height = M.rows();
    size_t width = M.cols();

    // Find max for normalization
    double v_max = 0;
    for (size_t i=1; i < height; ++i) {
        for (size_t j=1; j < width; ++j) {
            if (M(i,j) > v_max) {
                v_max = M(i,j);
            }
        }     
    }
    
    // Write results
    std::ofstream out("heatmap.ppm", std::ios::binary);
    out << "P6\n" << width-2 << " " << height-2 << "\n255\n"; 

     for (size_t i=1; i < height-1; ++i) {
        for (size_t j=1; j < width-1; ++j) {
            double v_norm = M(i,j) / v_max;
            uint8_t r = static_cast<uint8_t>(v_norm * 255);
            uint8_t g = static_cast<uint8_t>((1-std::abs(v_norm-0.5)*2) * 255);
            uint8_t b = static_cast<uint8_t>(255 - v_norm * 255);
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }

    out.close();
    std::cout << "Heatmap saved to heatmap.ppm\n";
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Error. Usage: \n";
        std::cerr << argv[0] << " nx ny n_iterations\n";
        return 1;
    }
    
    size_t nx = std::stoul(argv[1]);
    size_t ny = std::stoul(argv[2]);
    size_t n_iterations = std::stoul(argv[3]);

    MPI_Init(&argc, &argv);
    
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // local_nx is the number of rows assigned to each process
    // distribute reminder rows
    size_t local_nx;
    size_t local_nx_no_bound = nx / size;
    size_t remainder = nx % size;
    local_nx_no_bound += (rank < remainder ? 1 : 0);
    // Add boundary rows
    if (size == 1) {
        local_nx = local_nx_no_bound;
    } else if (rank == 0 || rank == size-1) {
        local_nx = local_nx_no_bound + 1;
    } else {
        local_nx = local_nx_no_bound + 2;
    }

    // SubMatrixes
    Matrix<double> M = Matrix<double>(local_nx, ny);
    Matrix<double> M_new = Matrix<double>(local_nx, ny);

    // Full Matrix
    Matrix<double> M_full = Matrix<double>(nx, ny);
    
    // Scatter info 
    std::vector<int> sendcounts, displs;
    if (rank == 0) {
        M_full = alloc_init_matrix(nx, ny);
        sendcounts.resize(size);
        displs.resize(size);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int rows = r < nx%size ? nx/size + 1 : nx/size;
            sendcounts[r] = rows * ny;
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    MPI_Scatterv(
            rank == 0 ? M_full.raw() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
            rank == 0 ? M.raw() : M.raw()+ny, local_nx*ny, MPI_DOUBLE, 0,
            MPI_COMM_WORLD
    );

    // Define process north and process south
    int proc_north = rank - 1 >= 0    ?  rank - 1 : MPI_PROC_NULL;
    int proc_south = rank + 1 <  size ?  rank + 1 : MPI_PROC_NULL;

    // Jacobi iterative algorithm
    for (size_t iteration = 0; iteration < n_iterations; ++iteration) {
        
        // Communicate boundary elements between processors
        // Send to north, receive from south
        MPI_Sendrecv(
            // Send first non-boundary row to process north
            M.raw() + ny,                   ny, MPI_DOUBLE, proc_north, 0,
            // Receive from process south the row to place in south boundary
            M.raw() + (local_nx - 1) * ny,  ny, MPI_DOUBLE, proc_south, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // Send to south, receive from north
        MPI_Sendrecv(
            // Send last non-boundary row to process south
            M.raw() + (local_nx - 2) * ny,  ny, MPI_DOUBLE, proc_south, 1,
            // Receive from process north the row to place in north boundary
            M.raw(),                        ny, MPI_DOUBLE, proc_north, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        // Jacobi iteration, skip boundary rows and columns
        for (size_t i = 1; i < local_nx - 1; ++i) {
            for (size_t j = 1; j < ny - 1; ++j) {
                M_new(i,j) = ( M(i-1,j) + M(i,j+1) + M(i+1,j) + M(i,j-1) )/4;
            }
        }
        
        // Boundary rows/cols remain unchanged
        for (size_t j = 0; j < ny; ++j) {
            M_new(0, j) = M(0, j);  // Top boundary row
            M_new(local_nx - 1, j) = M(local_nx - 1, j);  // Bottom boundary row
        }
        
        for (size_t i = 0; i < local_nx; ++i) {
            M_new(i, 0) = M(i, 0); //Left boundary column  
            M_new(i, ny - 1) = M(i, ny - 1); // Right boundary column
        }
        
        // In the next step, M is M_new
        std::swap(M, M_new);
    }

    // Collect submatrixes in full matrixes
    MPI_Gatherv(
        rank == 0 ? M.raw() : M.raw() + ny , local_nx_no_bound * ny, MPI_DOUBLE,
        M_full.raw(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    
    // Save heatmap of the results
    if (rank == 0) {
        heatmap(M_full);
    }

    MPI_Finalize();
    
    return 0; 
}
