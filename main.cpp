#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

double* generate_matrix(int n) {
    double* matrix = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = (double)std::rand() / 100000000;
        }
    }
    return matrix;
}

double* generate_vec(int n) {
    double* vec = new double[n];
    for (int i = 0; i < n; ++i) {
        vec[i] = (double)std::rand() / 10000000;
    }
    return vec;
}

double* transpose(double* matrix, int n) {
    double* res = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            res[i * n + j] = matrix[i * n + j];
            res[j * n + i] = matrix[j * n + i];
            std::swap(res[j * n + i], matrix[i * n + j]);
        }
    }
    return res;
}

double scalar_product(double* vec1, double* vec2, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

double* mul_matrix_on_vector(double* matrix, double* vec, int n) {
    double* res = new double[n];
    for (int i = 0; i < n; ++i) {
        res[i] = 0;
        for (int j = 0; j < n; ++j) {
            res[i] += matrix[i * n + j] * vec[j];
        }
    }
    return res;
}

double frob_norm(double* matrix, int n) {
    double norm = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            norm += matrix[i * n + j] * matrix[i * n + j];
        }
    }
    return std::sqrt(norm);
}

void print_vec(double* vec, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

double* BCG(double* matrix, double* right_part, int n) {
    double* r = right_part;
    double* p = new double[n];
    double* z = new double[n];
    double* s = new double[n];
    double* x = new double[n];
    double* transpose_matrix = transpose(matrix, n);
    for (int i = 0; i < n; ++i) {
        x[i] = 0;
        p[i] = r[i];
        z[i] = r[i];
        s[i] = r[i];
    }
    int iterations_cnt = 0;
    while (scalar_product(r, r, n) > 0.1) {
        double p_r_product = scalar_product(p, r, n);
        double* Az = mul_matrix_on_vector(matrix, z, n);
        double alpha = p_r_product / scalar_product(s, Az, n);
        std::cout << "alpha - " << alpha << std::endl;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * z[i];
        }
        double* As = mul_matrix_on_vector(transpose_matrix, s, n);
        for (int i = 0; i < n; ++i) {
            r[i] -= alpha * Az[i];
            p[i] -= alpha * As[i];
        }
        double beta = scalar_product(p, r, n) / p_r_product;
        for (int i = 0; i < n; ++i) {
            z[i] = r[i] + beta * z[i];
            s[i] = p[i] + beta * s[i];
        }
        ++iterations_cnt;
        if (iterations_cnt == 1000) break;
    }
    return x;
}

int main() {
    int n = 4;
    // double* matrix = new double[n * n];
    // double* matrix = generate_matrix(n);
    //double *matrix = generate_matrix(n);
    double matrix[16] = {100, 1, 1, 1,
                        1, 100, 1, 1,
                        1, 1, 100, 1,
                        1, 1, 1, 100};
    // double* solution = new double[n];
    double* solution = generate_vec(n);
    double* right_part = mul_matrix_on_vector(matrix, solution, n);
    double* mysolution = BCG(matrix, right_part, n);
    print_vec(solution, n);
    print_vec(mysolution, n);
    return 0;
}