#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <cmath>

void heatmap(std::vector<std::vector<double>> M, int height, int width) {
    // Find max
    double v_max = 0;
    for (int i=1; i < height; ++i) {
        for (int j=1; j < width; ++j) {
            if (M[i][j] > v_max) {
                v_max = M[i][j];
            }
        }      
    }
    
    // Write results
    std::ofstream out("heatmap.ppm", std::ios::binary);
    out << "P6\n" << width-2 << " " << height-2 << "\n255\n"; 

     for (int i=1; i < height-1; ++i) {
        for (int j=1; j < width-1; ++j) {
            double v_norm = M[i][j] / v_max;
            int r = static_cast<int>(v_norm * 255);
            int g = static_cast<int>((1-std::abs(v_norm-0.5)*2) * 255);
            int b = static_cast<int>(255 - v_norm * 255);
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }

    out.close();
    std::cout << "Heatmap saved to heatmap.ppm\n";
}

void jacobi_step(std::vector<std::vector<double>> &M, std::vector<std::vector<double>> &M_new, int n_rows, int n_cols) {
    for (int i = 1; i < n_rows-1; ++i) {
        for (int j = 1; j < n_cols-1; ++j) {
            M_new[i][j] = (M[i-1][j] + M[i][j+1] + M[i+1][j] + M[i][j-1])/4;
        }
    }
}

std::vector<std::vector<double>> alloc_init_matrix(int n_rows, int n_cols) {
    double v_max = 100.0;
    double v_inner = 0.5;
    
    // Allocate memory for matrix M
    std::vector<std::vector<double>> M(n_rows, std::vector<double>(n_cols, v_inner));
    
    // Set boundary conditions
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {  
            if (i == n_rows-1) {
                double q = static_cast<double>(j)/(n_cols-1);
                M[i][j] = v_max * (1.0-q); 
            } else if (j == 0) {
                double q = static_cast<double>(i)/(n_rows-1);
                M[n_rows-i-1][j] = v_max * (1.0-q);
            } else if (i == 0 || j == n_cols-1) {
                M[i][j] = 0;
            }
        }
    }

    return M;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Error. Usage: \n";
        std::cerr << argv[0] << " n_rows n_cols n_iterations\n";
        return 1;
    }

    int n_rows = std::atoi(argv[1]);
    int n_cols = std::atoi(argv[2]);
    int n_iterations = std::atoi(argv[3]);

    std::vector<std::vector<double>> M = alloc_init_matrix(n_rows, n_cols);
    std::vector<std::vector<double>> M_new(n_rows, std::vector<double>(n_cols));

    for (int t = 0; t < n_iterations ; ++t) {
        jacobi_step(M, M_new, n_rows, n_cols);
        std::swap(M, M_new);
    }

    heatmap(M_new, n_rows, n_cols);
    
    return 0;
}
