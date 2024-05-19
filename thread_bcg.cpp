#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>

using namespace std;

namespace {
    const double EPS = std::numeric_limits<float>::epsilon();
    const int ione = 1;
    const double dnone = -1;
    const double done = 1;
}

extern "C" { // Scales a vector by a constant
    void dscal_(const int *, const double *, double *, const int *);
}

extern "C" { // Vector copy
    void dcopy_(const int *, const double *, const int *, double *, const int *);
}

extern "C" { // Dot product
    double ddot_(const int *, const double *, const int *, const double *, const int *);
}

extern "C" { // Vector norm
    double dnrm2_(const int *, const double *, const int *);
}

extern "C" { // y := a*x + y
    void daxpy_(const int *, const double *, const double *, const int *, double *, const int *);
}

class Matrix_CSR { // CSR matrix format
    public:
        Matrix_CSR(vector<vector<double>> matrix) {
            int cnt = 0;
            for (int i = 0; i < matrix.size(); ++i) {
                for (int j = 0; j < matrix[0].size(); ++j) {
                    if (abs(matrix[i][j]) > EPS) {
                        values.emplace_back(matrix[i][j]);
                        columns.emplace_back(j);
                        ++cnt;
                    }
                }
                rows.emplace_back(cnt);
            }
        }

        void matvec(double *vec, double *res_vec) {
            for (int i = 0; i < rows.size(); ++i) {
                int prev = (i == 0) ? 0 : rows[i - 1];
                double cur = 0;
                for (int j = prev; j < rows[i]; ++j) {
                    cur += values[j] * vec[columns[j]];
                } 
                res_vec[i] = cur;
            }
        }

        vector<int> get_rows() {
            return rows;
        }

        vector<int> get_columns() {
            return columns;
        }

        vector<double> get_values() {
            return values;
        }

        int get_rows_size() {
            return rows.size();
        }

    private:
        vector<int> rows;
        vector<int> columns;
        vector<double> values;
};

void print_vec(int size, double* vec) {
    for (int i = 0; i < size; ++i) {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void matvec_worker(Matrix_CSR matrix, double *vec, double *res_vec, int start, int end) {
    vector<int> rows = matrix.get_rows();
    vector<int> columns = matrix.get_columns();
    vector<double> values = matrix.get_values();
    for (int i = start; i < end; ++i) {
        int prev = (i == 0) ? 0 : rows[i - 1];
        double cur = 0;
        for (int j = prev; j < rows[i]; ++j) {
            cur += values[j] * vec[columns[j]];
        } 
        res_vec[i] = cur;
    }
}

void matvec(Matrix_CSR matrix, double *vec, double *res_vec) {
    const int num_threads = std::thread::hardware_concurrency();
    vector<std::thread> threads;

    vector<int> rows = matrix.get_rows();
    int rows_per_thread = (rows.size() + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = std::min((i + 1) * rows_per_thread, static_cast<int>(rows.size()));
        threads.emplace_back(matvec_worker, matrix, vec, res_vec, start, end);
    }

    for (std::thread& thread : threads) {
        thread.join();
    }
}

template <typename T>
void BCG(T& matrix, T& transpose_matrix, void (*matvec)(const T, double *, double *), double *b, double *x, int size, int max_iter) {
    double* p = new double[size];
    double* z = new double[size];
    double* s = new double[size];
    double* r = new double[size];
    double* Ax = new double[size];
    double* Az = new double[size];
    double* As = new double[size];
    matvec(matrix, x, Ax); // Ax := A * x
    dcopy_(&size, b, &ione, r, &ione); // r := b
    daxpy_(&size, &dnone, Ax, &ione, r, &ione); // r := b - A * x
    dcopy_(&size, r, &ione, p, &ione); // p := r
    dcopy_(&size, r, &ione, z, &ione); // z := r
    dcopy_(&size, r, &ione, s, &ione); // s := r
    int iterations_cnt = 0;
    double p_r_product = ddot_(&size, p, &ione, r, &ione);
    while (dnrm2_(&size, r, &ione) > 0.1) { // check r
        matvec(matrix, z, Az); // Az := A * z
        double alpha = p_r_product / ddot_(&size, s, &ione, Az, &ione); // alpha := (p, r) / (s, Az)
        matvec(transpose_matrix, s, As); // As := AT * s
        daxpy_(&size, &alpha, z, &ione, x, &ione); // x := x + alpha * z
        double malpha = -1 * alpha;
        daxpy_(&size, &malpha, Az, &ione, r, &ione); // r := r - alpha * Az
        daxpy_(&size, &malpha, As, &ione, p, &ione); // p := p - alpha * As
        double prev_p_r_product = p_r_product;
        p_r_product = ddot_(&size, p, &ione, r, &ione);
        double beta = p_r_product / prev_p_r_product; // beta := (p, r) / (p,r)_prev
        dscal_(&size, &beta, z, &ione); // z := beta * z
        daxpy_(&size, &done, r, &ione, z, &ione); // z := beta * z + r
        dscal_(&size, &beta, s, &ione); // s := beta * s
        daxpy_(&size, &done, p, &ione, s, &ione); // s := beta * s + p
        ++iterations_cnt;
        if (iterations_cnt == max_iter) break; // check max_iter
    }
    std::cout << "iterations - " << iterations_cnt << std::endl;
    delete[] p;
    delete[] z;
    delete[] s;
    delete[] r;
    delete[] Ax;
    delete[] Az;
    delete[] As;
}

int main() {
    int size = 10000;
    srand(time(0));
    vector<vector<double>> matrix_(size, vector<double>(size, 0));
    for (int i = 0; i < size; ++i) {
        matrix_[i][i] = 4;
        if (i + 1 < size) {
            matrix_[i + 1][i] = 1;
            matrix_[i][i + 1] = 1;
        }
    } 
    Matrix_CSR matrix(matrix_);
    double *x = new double [size];
    double *b = new double [size];
    for (int i = 0; i < size; ++i) {
        x[i] =  i;
    }
    matrix.matvec(x, b);
    double *my_x = new double [size];
    for (int i = 0; i < size; ++i) {
        my_x[i] = 0;
    }
    auto begin = std::chrono::steady_clock::now();
    BCG(matrix, matrix, &matvec, b, my_x, size, 1000);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    for (int i = 0; i < size; ++i) {
        x[i] -= my_x[i];
    }
    cout << dnrm2_(&size, x, &ione) << endl;
}