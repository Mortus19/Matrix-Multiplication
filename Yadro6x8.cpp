﻿
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <string>
#include <omp.h>
#include <immintrin.h>

using namespace std;
constexpr double eps = 0.00000000001;

void my_micro_yadro_with_coppy(double* A, double* B, double* C, int offset_A, int offset_B, int offset_C, int small_z) {
    //A - 6 x zz
    //B - zz x 8
    //C - 6 x 8
    __m256d c11 = _mm256_load_pd(C);
    __m256d c12 = _mm256_load_pd(C + 4);

    __m256d c21 = _mm256_load_pd(C + offset_C);
    __m256d c22 = _mm256_load_pd(C + offset_C + 4);

    __m256d c31 = _mm256_load_pd(C + offset_C * 2);
    __m256d c32 = _mm256_load_pd(C + offset_C * 2 + 4);

    __m256d c41 = _mm256_load_pd(C + offset_C * 3);
    __m256d c42 = _mm256_load_pd(C + offset_C * 3 + 4);

    __m256d c51 = _mm256_load_pd(C + offset_C * 4);
    __m256d c52 = _mm256_load_pd(C + offset_C * 4 + 4);


    __m256d c61 = _mm256_load_pd(C + offset_C * 5);
    __m256d c62 = _mm256_load_pd(C + offset_C * 5 + 4);


    for (int i = 0; i < small_z; i++) {
        __m256d a1 = _mm256_set1_pd(*(A + i));
        __m256d a2 = _mm256_set1_pd(*(A + i + offset_A));

        __m256d b1 = _mm256_load_pd(B);
        __m256d b2 = _mm256_load_pd(B + 4);

        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c12 = _mm256_fmadd_pd(a1, b2, c12);

        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c22 = _mm256_fmadd_pd(a2, b2, c22);

        a1 = _mm256_set1_pd(*(A + i + offset_A * 2));
        a2 = _mm256_set1_pd(*(A + i + offset_A * 3));

        c31 = _mm256_fmadd_pd(a1, b1, c31);
        c32 = _mm256_fmadd_pd(a1, b2, c32);

        c41 = _mm256_fmadd_pd(a2, b1, c41);
        c42 = _mm256_fmadd_pd(a2, b2, c42);

        a1 = _mm256_set1_pd(*(A + i + offset_A * 4));
        a2 = _mm256_set1_pd(*(A + i + offset_A * 5));

        c51 = _mm256_fmadd_pd(a1, b1, c51);
        c52 = _mm256_fmadd_pd(a1, b2, c52);

        c61 = _mm256_fmadd_pd(a2, b1, c61);
        c62 = _mm256_fmadd_pd(a2, b2, c62);
        B += offset_B;
    }

    _mm256_store_pd(C, c11);
    _mm256_store_pd(C + 4, c12);

    _mm256_store_pd(C + offset_C, c21);
    _mm256_store_pd(C + offset_C + 4, c22);

    _mm256_store_pd(C + offset_C * 2, c31);
    _mm256_store_pd(C + offset_C * 2 + 4, c32);

    _mm256_store_pd(C + offset_C * 3, c41);
    _mm256_store_pd(C + offset_C * 3 + 4, c42);

    _mm256_store_pd(C + offset_C * 4, c51);
    _mm256_store_pd(C + offset_C * 4 + 4, c52);

    _mm256_store_pd(C + offset_C * 5, c61);
    _mm256_store_pd(C + offset_C * 5 + 4, c62);

}

class Matrix {
    /*
    mx_size - размер , который является степенью двойки
    нужен для того, чтобы нормально работало перемножение матриц в алгоритме Штрассена
    */
    int size;
    int mx_size;
    vector<double> m;
public:
    Matrix(int _size = 0) {
        size = _size;
        mx_size = size;
        m.assign(mx_size * mx_size, 0);
    }

    Matrix(int _size, vector<double>& _m) {
        size = _size;
        mx_size = size;
        m.assign(mx_size * mx_size, 0);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                m[i * mx_size + j] = _m[i * size + j];
            }
        }
    }

    Matrix(const Matrix& B) {
        size = B.size;
        mx_size = B.mx_size;
        m = B.m;
    }

    void resize(int _size) {
        size = _size;
        mx_size = size;
       
        m.assign(mx_size * mx_size, 0);
    }

    void clear() {
        size = 0;
        mx_size = 0;
        m.clear();
    }

    double& operator()(const int i, const int j) {
        return m[i * size + j];
    }

    Matrix operator+(const Matrix& B) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] + B.m[i * mx_size + j];
            }
        }
        return ans;
    }

    Matrix operator-(const Matrix& B) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] - B.m[i * mx_size + j];
            }
        }
        return ans;
    }

    Matrix operator*(double value) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] * value;
            }
        }
        return ans;
    }

    bool operator==(const Matrix& B) {
        if (&B == this) return true;
        if (size == B.size && B.mx_size == mx_size) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    if (abs(m[i * mx_size + j] - B.m[i * mx_size + j]) > eps)
                        return false;
                }
            }
            return true;
        }
        return false;
    }

    bool operator!=(const Matrix& B) {
        return !((*this) == B);
    }


    Matrix& operator=(const Matrix& B) {
        if (&B == this) return *this;
        this->clear();
        size = B.size;
        mx_size = B.mx_size;
        m = B.m;
        return *this;
    }

    friend Matrix mult(Matrix& A, Matrix& B) {
        const int N = A.mx_size;
        Matrix C(A.size);
        for (int i = 0; i < A.size; i++)
            for (int j = 0; j < A.size; j++)
                C.m[i * N + j] = 0;
        int x, y, z;
        
        #pragma omp parallel for
        for (int i = 0; i < A.size; i++) {
            for (int k = 0; k < A.size; k++) {
                #pragma omp simd
                for (int j = 0; j < A.size; j++) {
                    C.m[i * N + j] += A.m[i * N + k] * B.m[k * N + j];
                }
            }
        }
        return C;
    }

    

    friend void my_micro_yadro2(const Matrix& A, double* B, double* C, int offset, int offset_A, int offset_B, int offset_C, int small_z) {
        //A - 6 x zz
        //B - zz x 8
        //C - 6 x 8
        C += offset_C;
        B += offset_B;
        __m256d c11 = _mm256_load_pd(C);
        __m256d c12 = _mm256_load_pd(C + 4);

        __m256d c21 = _mm256_load_pd(C + offset);
        __m256d c22 = _mm256_load_pd(C + offset + 4);

        __m256d c31 = _mm256_load_pd(C + offset * 2);
        __m256d c32 = _mm256_load_pd(C + offset * 2 + 4);
        
        __m256d c41 = _mm256_load_pd(C + offset * 3);
        __m256d c42 = _mm256_load_pd(C + offset * 3 + 4);

        __m256d c51 = _mm256_load_pd(C + offset * 4);
        __m256d c52 = _mm256_load_pd(C + offset * 4 + 4);


        __m256d c61 = _mm256_load_pd(C + offset * 5);
        __m256d c62 = _mm256_load_pd(C + offset * 5 + 4);


        for (int i = 0; i < small_z; i++) {
            __m256d a1 = _mm256_set1_pd(A.m[offset_A + i]);
            __m256d a2 = _mm256_set1_pd(A.m[offset_A + i + offset]);

            __m256d b1 = _mm256_load_pd(B);
            __m256d b2 = _mm256_load_pd(B + 4);
            

            c11 = _mm256_fmadd_pd(a1, b1, c11);
            c12 = _mm256_fmadd_pd(a1, b2, c12);
            
            c21 = _mm256_fmadd_pd(a2, b1, c21);
            c22 = _mm256_fmadd_pd(a2, b2, c22);
            
            a1 = _mm256_set1_pd(A.m[offset_A + i + offset*2]);
            a2 = _mm256_set1_pd(A.m[offset_A + i + offset*3]);

            c31 = _mm256_fmadd_pd(a1, b1, c31);
            c32 = _mm256_fmadd_pd(a1, b2, c32);

            c41 = _mm256_fmadd_pd(a2, b1, c41);
            c42 = _mm256_fmadd_pd(a2, b2, c42);

            a1 = _mm256_set1_pd(A.m[offset_A + i + offset * 4]);
            a2 = _mm256_set1_pd(A.m[offset_A + i + offset * 5]);

            c51 = _mm256_fmadd_pd(a1, b1, c51);
            c52 = _mm256_fmadd_pd(a1, b2, c52);

            c61 = _mm256_fmadd_pd(a2, b1, c61);
            c62 = _mm256_fmadd_pd(a2, b2, c62);
            B += offset;
        }

        _mm256_store_pd(C, c11);
        _mm256_store_pd(C + 4, c12);

        _mm256_store_pd(C + offset, c21);
        _mm256_store_pd(C + offset + 4, c22);

        _mm256_store_pd(C + offset * 2, c31);
        _mm256_store_pd(C + offset * 2 + 4, c32);

        _mm256_store_pd(C + offset * 3, c41);
        _mm256_store_pd(C + offset * 3 + 4, c42);

        _mm256_store_pd(C + offset * 4, c51);
        _mm256_store_pd(C + offset * 4 + 4, c52);

        _mm256_store_pd(C + offset * 5, c61);
        _mm256_store_pd(C + offset * 5 + 4, c62);

    }


    friend Matrix mult_block(const Matrix& A, const Matrix& B) {

        int N = A.size;
        constexpr int m = 6 * 8;//x
        constexpr int p = 8 * 8 * 2;//y
        constexpr int n = 8 * 8 * 4;//z
        constexpr int small_x = 6;
        constexpr int small_y = 8;
        constexpr int small_z = 64;

        Matrix ans(N);
        double sizekb = m * n + n * p + m * p;
        //sizekb = small_x * small_y + small_x * small_z + small_z * small_y;
        sizekb *= 8;
        sizekb /= 1024;
        int offset = A.mx_size;
        //есть еще идея избавиться от min , но это позже
        //cout << sizekb << '\n';

        
        #pragma omp parallel for
        for (int y = 0; y < N; y += p)
        {
            double AA[small_x * small_z];
            double BB[small_z * small_y];
            double CC[small_x * small_y];
            for (int z = 0; z < N; z += n)
                for (int x = 0; x < N; x += m)

                    for (int zz = z; zz < min(N, z + n); zz += small_z)
                        for (int yy = y; yy < min(N, y + p); yy += small_y)
                            for (int xx = x; xx < min(N, x + m); xx += small_x)
                            {

                                if (xx + small_x >= N || yy + small_y >= N || zz + small_z >= N) {
                                    //just naive
                                    for (int i = xx; i < min(xx + small_x, N); i++) {
                                        int C_const = i * A.mx_size;
                                        for (int k = zz; k < min(zz + small_z, N); k++) {
                                            int A_const = C_const + k;
                                            int B_const = k * A.mx_size;
                                            #pragma omp simd
                                            for (int j = yy; j < min(yy + small_y, N); j++) {
                                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                                                //cout << ans.m[C_const + j] << '\n';
                                            }
                                        }
                                    }

                                }
                                else {

                                   /* my_micro_yadro2(A, (double*)B.m.data(), (double*)ans.m.data(), offset,
                                        xx * A.mx_size + zz, zz * A.mx_size + yy, xx * A.mx_size + yy, small_z);*/
                                    for (int i = 0; i < small_x; i++) {
                                        memcpy(AA + i * small_z, A.m.data() + (xx + i) * A.mx_size + zz, sizeof(double) * small_z);
                                        memcpy(CC + i * small_y, ans.m.data() + (xx + i) * A.mx_size + yy, sizeof(double) * small_y);
                                    }
                                    for (int i = 0; i < small_z; i++) {
                                        memcpy(BB + i * small_y, B.m.data() + (zz + i) * A.mx_size + yy, sizeof(double) * small_y);
                                    }
                                    my_micro_yadro_with_coppy(AA, BB, CC, small_z, small_y, small_y, small_z);
                                    for (int i = 0; i < small_x; i++) {
                                        memcpy(ans.m.data() + (xx + i) * A.mx_size + yy, CC + i * small_y, sizeof(double) * small_y);
                                    }
                                }

                            }
        }

        return ans;
    }
    

    friend void read_bin(const string& input, Matrix& A, Matrix& B) {
        A.clear();
        B.clear();
        ifstream in(input, ios::binary | ios::in);
        in.read((char*)&A.size, sizeof(A.size));
        A.mx_size = A.size;
        vector<double> _m(A.size * A.size);
        in.read(reinterpret_cast<char*>(&_m[0]), A.size * A.size * sizeof(_m[0]));
        A.m.assign(A.mx_size * A.mx_size, 0);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                A.m[i * A.mx_size + j] = _m[i * A.size + j];
            }
        }
        in.read((char*)&B.size, sizeof(B.size));
        B.mx_size = A.mx_size;
        in.read(reinterpret_cast<char*>(&_m[0]), A.size * A.size * sizeof(_m[0]));
        B.m.assign(B.mx_size * B.mx_size, 0);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                B.m[i * B.mx_size + j] = _m[i * B.size + j];
            }
        }
    }

    friend void read_txt(const string& input, Matrix& A, Matrix& B) {
        A.clear();
        B.clear();
        ifstream in(input);
        in >> A >> B;
    }

    friend void output_bin(const string& output, Matrix& A) {
        ofstream out(output, ios::binary);
        out.write(reinterpret_cast<char*>(&A.size), sizeof(A.size));
        vector<double> _m(A.size * A.size);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                _m[i * A.size + j] = A.m[A.mx_size * i + j];
            }
        }
        out.write(reinterpret_cast<char*>(&_m[0]), A.size * A.size * sizeof(_m[0]));
    }

    friend ostream& operator<<(ostream& out, const Matrix& obj) {
        out << obj.size << '\n';
        for (int i = 0; i < obj.size; ++i) {
            for (int j = 0; j < obj.size; ++j) {
                out << obj.m[i * obj.mx_size + j] << ' ';
            }
            out << '\n';
        }
        return out;
    }

    friend istream& operator>>(istream& in, Matrix& obj) {
        obj.clear();
        in >> obj.size;
        obj.mx_size = 1;
        while (obj.mx_size < obj.size) {
            obj.mx_size *= 2;
        }
        obj.m.resize(obj.mx_size * obj.mx_size);
        for (int i = 0; i < obj.size; ++i) {
            for (int j = 0; j < obj.size; ++j) {
                in >> obj.m[i * obj.mx_size + j];
            }
        }
        return in;
    }

};

mt19937 random_generator(chrono::steady_clock::now().time_since_epoch().count());


int main(int argc, char* argv[]) {

    const string input_file_bin = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.bin";
    const string input_file_txt = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.txt";
    const string output_time = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output.txt";
    const string my_ans = "C:/Users/Zver/Source/Repos/Project4/x64/Release/my_ans.bin";


    Matrix A, B, C,C2;
    read_bin(input_file_bin, A, B);
    auto start = chrono::high_resolution_clock::now();
    C = mult_block(A, B);
    //C2 = mult(A, B);
    /*if (C != C2) {
        cout << "BOBBA\n";
    }*/
    ofstream out(output_time);
    auto end = chrono::high_resolution_clock::now();
    
    
    chrono::duration<double> duration = (end - start);
    duration *= 1000.0 * 1000.0;
    out.precision(10);
    out << fixed << duration.count() << '\n';
    output_bin(my_ans, C);
    return 0;
}