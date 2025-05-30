#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>
#include <string>
#include <mpi>

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

void jacobi_step(Matrix<double> &M, Matrix<double> &M_new) {
    for (size_t i = 1; i < M.rows()-1; ++i) {
        for (size_t j = 1; j < M.cols()-1; ++j) {
            M_new(i,j) = ( M(i-1,j) + M(i,j+1) + M(i+1,j) + M(i,j-1) )/4;
        }
    }
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

    Matrix<double> M = alloc_init_matrix(nx, ny);
    Matrix<double> M_new(nx, ny, 0.0);

    for (size_t t = 0; t < n_iterations ; ++t) {
        jacobi_step(M, M_new);
        std::swap(M, M_new);
    }

    heatmap(M_new);
    
    return 0;
}
