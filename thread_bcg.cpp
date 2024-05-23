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

double ddot_worker(int start, int end, double* vec1, double* vec2) {
    double res = 0;
    for (int i = start; i < end; ++i) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

double ddot(int size, double* vec1, double* vec2) {
    const int num_threads = std::thread::hardware_concurrency();
    int chunk_size = size / num_threads;
    int leftover = size % num_threads;

    std::vector<std::thread> threads;
    std::vector<double> results(num_threads, 0.0);
    int start = 0;

    for (int i = 0; i < num_threads; ++i) {
        int end = start + chunk_size + (i < leftover ? 1 : 0);
        threads.emplace_back([&](int s, int e, double* v1, double* v2, double& result) {
            result = ddot_worker(s, e, v1, v2);
        }, start, end, std::ref(vec1), std::ref(vec2), std::ref(results[i]));
        start = end;
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    double res = 0;
    for (double r : results) {
        res += r;
    }
    return res;
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

        vector<int> get_rows() const {
            return rows;
        }

        vector<int> get_columns() const {
            return columns;
        }

        vector<double> get_values() const {
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

void matvec_worker(const Matrix_CSR& matrix, double *vec, double *res_vec, int start, int end) {
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

void matvec(const Matrix_CSR& matrix, double *vec, double *res_vec) {
    const int num_threads = std::thread::hardware_concurrency();
    vector<std::thread> threads;

    vector<int> rows = matrix.get_rows();
    int rows_per_thread = (rows.size() + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int start = i * rows_per_thread;
        int end = min((i + 1) * rows_per_thread, static_cast<int>(rows.size()));
        threads.emplace_back(matvec_worker, std::ref(matrix), vec, res_vec, start, end);
    }

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void daxpy_thread(double alpha, double* x, double* y, int size, int done) {
    daxpy_(&size, &alpha, x, &ione, y, &ione);
}

template <typename T>
void BCG(const T& matrix, const T& transpose_matrix, double *b, double *x, int size, int max_iter) {
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
    double p_r_product = ddot(size, p, r);
    while (dnrm2_(&size, r, &ione) > 0.1) { // check r
        matvec(matrix, z, Az); // Az := A * z
        double alpha = p_r_product / ddot(size, s, Az); // alpha := (p, r) / (s, Az)
        double malpha = -1 * alpha;
        matvec(transpose_matrix, s, As); // As := AT * s

        std::thread thread_x(daxpy_thread, alpha, std::ref(z), std::ref(x), size, done);
        std::thread thread_r(daxpy_thread, malpha, std::ref(Az), std::ref(r), size, done);
        std::thread thread_p(daxpy_thread, malpha, std::ref(As), std::ref(p), size, done);
        thread_x.join();
        thread_r.join();
        thread_p.join();
        double prev_p_r_product = p_r_product;
        p_r_product = ddot(size, p, r);
        double beta = p_r_product / prev_p_r_product; // beta := (p, r) / (p,r)_prev

        std::thread thread_z([&]() { // z := beta * z + r
            dscal_(&size, &beta, z, &ione); // z := beta * z
            daxpy_(&size, &done, r, &ione, z, &ione); // z := beta * z + r
        });

        std::thread thread_s([&]() { // s := beta * s + p
            dscal_(&size, &beta, s, &ione); // s := beta * s
            daxpy_(&size, &done, p, &ione, s, &ione); // s := beta * s + p
        });

        thread_z.join();
        thread_s.join();

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
    int size = 20000;
    srand(time(0));
    vector<vector<double>> matrix_(size, vector<double>(size, 0));

    for (int i = 0; i < size; ++i) {
        matrix_[i][i] = rand();
        // matrix_[i][i] = 4;
        // if (i + 1 < size) {
        //     matrix_[i + 1][i] = 1;
        //     matrix_[i][i + 1] = 1;
        // }
        if (i + 1 < size) {
            matrix_[i + 1][i] = rand() / 100000;
            matrix_[i][i + 1] = rand() / 100000;
        }
        // if (i + 2 < size) {
        //     matrix_[i + 2][i] = rand() / 100000;
        //     matrix_[i][i + 2] = rand() / 100000;
        // }
    } 

    // for (int i = 0; i < size; ++i) {
    //     matrix_[i][i] = 4;
    //     if (i + 1 < size) {
    //         matrix_[i + 1][i] = 1;
    //         matrix_[i][i + 1] = 1;
    //     }
    // } 
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
    BCG(matrix, matrix, b, my_x, size, 1000);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    for (int i = 0; i < size; ++i) {
        x[i] -= my_x[i];
    }
    cout << "error - " << dnrm2_(&size, x, &ione) << endl;
}