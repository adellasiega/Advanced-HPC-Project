#include <iostream>

void jacobi_evaluation(double **M, int n_rows, int n_cols){
    for (int i = 1; i < n_rows-1; ++i) {
        for (int j = 1; j < n_cols-1; ++j) {
            M[i][j] = (M[i-1][j] + M[i][j+1] + M[i+1][j] + M[i][j-1])/4;
        }
    }
}

int main() {
    int n_rows = 10;
    int n_cols = 15;
    double v_max_init = 100.0;

    // Allocate memory for row pointers
    double** M = new double*[n_rows];

    // Allocate memory for each row
    for (int i = 0; i < n_rows; ++i) {
        M[i] = new double[n_cols];
    }

    // Initialize the M, boundary conditions
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {  
            if (i == n_rows-1) {
                double q = static_cast<double>(j)/(n_cols-1);
                M[i][j] = v_max_init * (1.0-q); 
            } else if (j == 0) {
                double q = static_cast<double>(i)/(n_rows-1);
                M[n_rows-i-1][j] = v_max_init * (1.0-q);
            } else {
                M[i][j] = 0;
            }
        }
    }

    // Perform Jacobi iteration
    jacobi_evaluation(M, n_rows, n_cols);

    // Print M
    std::cout << "Matrix:" << std::endl;
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Deallocate memory
    for (int i = 0; i < n_rows; ++i) {
        delete[] M[i];  // delete each row
    }
    delete[] M;  // delete the array of row pointers

    return 0;
}
