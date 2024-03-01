
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <string>
#include <omp.h>
#include <immintrin.h>

using namespace std;
constexpr double eps = 0.00000000001;

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
        mx_size = 1;
        while (mx_size < size) {
            mx_size *= 2;
        }
        m.assign(mx_size * mx_size, 0);
    }

    Matrix(int _size, vector<double> &_m) {
        size = _size;
        mx_size = 1;
        while (mx_size < size) {
            mx_size *= 2;
        }
        m.assign(mx_size * mx_size, 0);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                m[i * mx_size + j] = _m[i * size + j];
            }
        }
    }

    Matrix(const Matrix &B) {
        size = B.size;
        mx_size = B.mx_size;
        m = B.m;
    }

    void resize(int _size) {
        size = _size;
        mx_size = 1;
        while (mx_size < size) {
            mx_size *= 2;
        }
        m.assign(mx_size * mx_size, 0);
    }

    void clear() {
        size = 0;
        mx_size = 0;
        m.clear();
    }

    double &operator()(const int i, const int j) {
        return m[i * size + j];
    }

    Matrix operator+(const Matrix &B) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] + B.m[i * mx_size + j];
            }
        }
        return ans;
    }

    Matrix operator-(const Matrix &B) {
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

    bool operator==(const Matrix &B) {
        if (&B == this) return true;
        if (size = B.size && B.mx_size == mx_size) {
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

    bool operator!=(const Matrix &B) {
        return !((*this) != B);
    }


    Matrix &operator=(const Matrix &B) {
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

    friend Matrix mult_Strassen(const Matrix &A, const Matrix &B, int size) {

        if (size <= 16) {
            //Наивное умножение
            Matrix C(size);
            for (int i = 0; i < size; i++) {
                for (int k = 0; k < size; k++) {
                    for (int j = 0; j < size; j++) {
                        C.m[i * C.mx_size + j] += A.m[i * A.mx_size + k] * B.m[k * B.mx_size + j];
                    }
                }
            }
            return C;
        }

        int new_size = size / 2;
        Matrix A11(new_size);
        Matrix A12(new_size);
        Matrix A21(new_size);
        Matrix A22(new_size);
        Matrix B11(new_size);
        Matrix B12(new_size);
        Matrix B21(new_size);
        Matrix B22(new_size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i < new_size && j < new_size) {
                    A11.m[i * new_size + j] = A.m[i * size + j];
                    B11.m[i * new_size + j] = B.m[i * size + j];
                } else if (i < new_size && j >= new_size) {
                    A12.m[i * new_size + (j - new_size)] = A.m[i * size + j];
                    B12.m[i * new_size + (j - new_size)] = B.m[i * size + j];
                } else if (i >= new_size && j < new_size) {
                    A21.m[(i - new_size) * new_size + j] = A.m[i * size + j];
                    B21.m[(i - new_size) * new_size + j] = B.m[i * size + j];
                } else {
                    A22.m[(i - new_size) * new_size + (j - new_size)] = A.m[i * size + j];
                    B22.m[(i - new_size) * new_size + (j - new_size)] = B.m[i * size + j];
                }

            }
        }
        Matrix S1 = A21 + A22;
        Matrix S2 = S1 - A11;
        Matrix S3 = A11 - A21;
        Matrix S4 = A12 - S2;
        Matrix S5 = B12 - B11;
        Matrix S6 = B22 - S5;
        Matrix S7 = B22 - B12;
        Matrix S8 = S6 - B21;

        Matrix P1 = mult_Strassen(S2, S6, new_size);
        Matrix P2 = mult_Strassen(A11, B11, new_size);
        Matrix P3 = mult_Strassen(A12, B21, new_size);
        Matrix P4 = mult_Strassen(S3, S7, new_size);
        Matrix P5 = mult_Strassen(S1, S5, new_size);
        Matrix P6 = mult_Strassen(S4, B22, new_size);
        Matrix P7 = mult_Strassen(A22, S8, new_size);

        Matrix T1 = P1 + P2;
        Matrix T2 = T1 + P4;

        Matrix C11 = P2 + P3;
        Matrix C12 = T1 + P5 + P6;
        Matrix C21 = T2 - P7;
        Matrix C22 = T2 + P5;

        Matrix C(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double &val = C.m[i * C.mx_size + j];
                if (i < new_size && j < new_size) {
                    val = C11.m[i * C11.mx_size + j];
                } else if (i < new_size && j >= new_size) {
                    val = C12.m[i * C12.mx_size + (j - new_size)];
                } else if (i >= new_size && j < new_size) {
                    val = C21.m[(i - new_size) * C21.mx_size + (j)];
                } else {
                    val = C22.m[(i - new_size) * C22.mx_size + (j - new_size)];
                }
            }
        }
        return C;
    }

    friend Matrix mult_Strassen(const Matrix &A, const Matrix &B) {
        Matrix C = mult_Strassen(A, B, A.mx_size);
        C.size = A.size;
        return C;
    }

    inline friend void my_micro_yadro(const Matrix& A , double* B , double* C, int offset , int offset_A, int offset_B, int offset_C) {
        //A - 8 x 2
        //B - 2 x 16
        //C - 8 x 16
        C += offset_C;
        B += offset_B;
        __m256d b11 = _mm256_load_pd(B);
        __m256d b12 = _mm256_load_pd(B+4);
        __m256d b13 = _mm256_load_pd(B+8);
        __m256d b14 = _mm256_load_pd(B+12);
        B += offset;
        __m256d b21 = _mm256_load_pd(B);
        __m256d b22 = _mm256_load_pd(B + 4);
        __m256d b23 = _mm256_load_pd(B + 8);
        __m256d b24 = _mm256_load_pd(B + 12);

        for (int i = 0; i < 8; i++) {
            __m256d a1 = _mm256_set1_pd(A.m[offset_A]);
            __m256d a2 = _mm256_set1_pd(A.m[offset_A + 1]);

            __m256d c1 = _mm256_load_pd(C);
            __m256d c2 = _mm256_load_pd(C+4);
            __m256d c3 = _mm256_load_pd(C+8);
            __m256d c4 = _mm256_load_pd(C+12);

            c1 = _mm256_fmadd_pd(a1, b11, c1);
            c2 = _mm256_fmadd_pd(a1, b12, c2);
            c3 = _mm256_fmadd_pd(a1, b13, c3);
            c4 = _mm256_fmadd_pd(a1, b14, c4);

            c1 = _mm256_fmadd_pd(a2, b21, c1);
            c2 = _mm256_fmadd_pd(a2, b22, c2);
            c3 = _mm256_fmadd_pd(a2, b23, c3);
            c4 = _mm256_fmadd_pd(a2, b24, c4);

            _mm256_store_pd(C, c1);
            _mm256_store_pd(C+4, c2);
            _mm256_store_pd(C+8, c3);
            _mm256_store_pd(C+12, c4);
            offset_A += offset;
            C += offset;
        }
    }
    inline friend void my_micro_yadro2(const Matrix& A, double* B, double* C, int offset, int offset_A, int offset_B, int offset_C, int small_z) {
        //A - 6 x zz
        //B - zz x 8
        //C - 6 x 8
        C += offset_C;
        B += offset_B;
        __m256d c11 = _mm256_load_pd(C);
        __m256d c12 = _mm256_load_pd(C + 4);

        __m256d c21 = _mm256_load_pd(C + (offset));
        __m256d c22 = _mm256_load_pd(C + (offset + 4));
        
        __m256d c31 = _mm256_load_pd(C + (offset * 2));
        __m256d c32 = _mm256_load_pd(C + (offset * 2 + 4));
        
        __m256d c41 = _mm256_load_pd(C + (offset * 3));
        __m256d c42 = _mm256_load_pd(C + (offset * 3 + 4));
        
        __m256d c51 = _mm256_load_pd(C + (offset * 4));
        __m256d c52 = _mm256_load_pd(C + (offset * 4 + 4));

        __m256d c61 = _mm256_load_pd(C + (offset * 5));
        __m256d c62 = _mm256_load_pd(C + (offset * 5 + 4));
 

        for (int i = 0; i < small_z; i++) {
            __m256d a1 = _mm256_set1_pd(A.m[offset_A + i]);
            __m256d a2 = _mm256_set1_pd(A.m[offset_A + i + offset]);
            __m256d b1 = _mm256_load_pd(B);
            __m256d b2 = _mm256_load_pd(B + 4);
        
            c11 = _mm256_fmadd_pd(a1, b1, c11);
            c12 = _mm256_fmadd_pd(a1, b2, c12);
            
            c21 = _mm256_fmadd_pd(a2, b1, c21);
            c22 = _mm256_fmadd_pd(a2, b2, c22);

            a1 = _mm256_set1_pd(A.m[offset_A + i + offset * 2]);
            a2 = _mm256_set1_pd(A.m[offset_A + i + offset * 3]);
            
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
        
        _mm256_store_pd(C + (offset), c21);
        _mm256_store_pd(C + (offset + 4), c22);
        
        _mm256_store_pd(C + (offset * 2), c31);
        _mm256_store_pd(C + (offset * 2 + 4), c32);

        _mm256_store_pd(C + (offset * 3), c41);
        _mm256_store_pd(C + (offset * 3 + 4), c42);

        _mm256_store_pd(C + (offset * 4), c51);
        _mm256_store_pd(C + (offset * 4 + 4), c52);

        _mm256_store_pd(C + (offset * 5), c61);
        _mm256_store_pd(C + (offset * 5 + 4), c62);

    }

    friend Matrix mult_block(const Matrix& A, const Matrix& B) {

        int N = A.size;
        constexpr int m = 8 * 6;//x
        constexpr int p = 8 * 8 * 4;//y
        constexpr int n = 8 * 8;//z
        constexpr int small_x = 6;
        constexpr int small_y = 8;
        constexpr int small_z = 16;

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
            for (int x = 0; x < N; x += m)
                for (int z = 0; z < N; z += n)

                    for (int zz = z; zz < min(N, z + n); zz += small_z)
                        for (int xx = x; xx < min(N, x + m); xx += small_x)
                            for (int yy = y; yy < min(N, y + p); yy += small_y)
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
                                            }
                                        }
                                    }
                                }
                                else {
                                    my_micro_yadro2(A, (double*)B.m.data(), (double*)ans.m.data(), offset,
                                        xx * A.mx_size + zz, zz * A.mx_size + yy, xx * A.mx_size + yy,small_z);
                                }

                            }
                        

                    
                
        return ans;
    }
    friend void add(Matrix &A, Matrix &B, Matrix &C, int ax, int ay, int bx, int by, int cx, int cy, int cur_size) {
        for (int i = 0; i < cur_size; i++) {
            for (int j = 0; j < cur_size; j++) {
                C.m[(i + cx) * C.mx_size + (j + cy)] =
                        A.m[(i + ax) * A.mx_size + (j + ay)] + B.m[(i + bx) * B.mx_size + (j + by)];
            }
        }
    }

    friend void sub(Matrix &A, Matrix &B, Matrix &C, int ax, int ay, int bx, int by, int cx, int cy, int cur_size) {

        for (int i = 0; i < cur_size; i++) {
            for (int j = 0; j < cur_size; j++) {
                C.m[(i + cx) * C.mx_size + (j + cy)] =
                        A.m[(i + ax) * A.mx_size + (j + ay)] - B.m[(i + bx) * B.mx_size + (j + by)];
            }
        }

    }

    friend void
    Mult_Strassen_Memory_efficient(Matrix &A, Matrix &B, Matrix &C, int ax, int ay, int bx, int by, int cx, int cy,
                                   int cur_size) {

        if (cur_size <= 16) {
            //Наивное умножение
            for (int i = 0; i < cur_size; i++) {
                for (int j = 0; j < cur_size; j++) {
                    C.m[(i + cx) * C.mx_size + (j + cy)] = 0;
                }
            }
            for (int i = 0; i < cur_size; i++) {
                for (int k = 0; k < cur_size; k++) {
                    for (int j = 0; j < cur_size; j++) {
                        C.m[(i + cx) * C.mx_size + (j + cy)] +=
                                A.m[(i + ax) * A.mx_size + (k + ay)] * B.m[(k + bx) * B.mx_size + (j + by)];
                    }
                }
            }
            return;
        }

        int new_size = cur_size / 2;

        //A11 -> [ax , ax + new_size-1][ay, ay + new_size-1]
        //A12 -> [ax , ax + new_size-1][ay+new_size, ay + 2 * new_size-1]
        //A21 -> [ax+new_size , ax + 2 * new_size-1][ay, ay + new_size-1]
        //A22 -> [ax+new_size , ax + 2 * new_size-1][ay+new_size, ay + 2 * new_size-1]

        //1.S3 = A11 - A21 loc C11
        sub(A, A, C, ax, ay, ax + new_size, ay, cx, cy, new_size);

        //2.S1 = A21 + A22 loc A21
        add(A, A, A, ax + new_size, ay, ax + new_size, ay + new_size, ax + new_size, ay, new_size);

        //3.T1 = B12 − B11 loc C22
        sub(B, B, C, bx, by + new_size, bx, by, cx + new_size, cy + new_size, new_size);

        //4.T3 = B22 − B12 loc B12
        sub(B, B, B, bx + new_size, by + new_size, bx, by + new_size, bx, by + new_size, new_size);

        //5.P7 = S3 * T3  loc C21 -> C11 * B12 loc C21
        Mult_Strassen_Memory_efficient(C, B, C, cx, cy, bx, by + new_size, cx + new_size, cy, new_size);

        //6.S2 = S1 − A11 loc B12 -> A21 - A11 loc B12
        sub(A, A, B, ax + new_size, ay, ax, ay, bx, by + new_size, new_size);

        //7.P1 = A11 * B11 loc C11
        Mult_Strassen_Memory_efficient(A, B, C, ax, ay, bx, by, cx, cy, new_size);

        //8.T2 = B22 − T1 loc B11 -> B22 - C22 loc B11
        sub(B, C, B, bx + new_size, by + new_size, cx + new_size, cy + new_size, bx, by, new_size);

        //9.P5 = S1 * T1 loc A11 -> A21 * C22 loc A11
        Mult_Strassen_Memory_efficient(A, C, A, ax + new_size, ay, cx + new_size, cy + new_size, ax, ay, new_size);

        //10.T4 = T2 − B21 loc C22 -> B11 - B21 loc C22
        sub(B, B, C, bx, by, bx + new_size, by, cx + new_size, cy + new_size, new_size);

        //11.P4 = A22 * T4 loc A21 -> A22 * C22 loc A21
        Mult_Strassen_Memory_efficient(A, C, A, ax + new_size, ay + new_size, cx + new_size, cy + new_size,
                                       ax + new_size, ay, new_size);

        //12.S4 = A12 − S2 loc C22 -> A12 - B12 loc C22
        sub(A, B, C, ax, ay + new_size, bx, by + new_size, cx + new_size, cy + new_size, new_size);

        //13.P6 = S2*T2 loc C12 -> B12 * B11 loc C12
        Mult_Strassen_Memory_efficient(B, B, C, bx, by + new_size, bx, by, cx, cy + new_size, new_size);

        //14.U2 = P1 + P6 loc C12 -> C11 + C12 loc C12
        add(C, C, C, cx, cy, cx, cy + new_size, cx, cy + new_size, new_size);

        //15.U3 = U2 + P7 loc C21 -> C12 + C21 loc C21
        add(C, C, C, cx, cy + new_size, cx + new_size, cy, cx + new_size, cy, new_size);

        //16.P3 = S4 * B22 loc B11 -> C22 * B22 loc B11
        Mult_Strassen_Memory_efficient(C, B, B, cx + new_size, cy + new_size, bx + new_size, by + new_size, bx, by,
                                       new_size);

        //17.U7 = U3 + P5 loc C22 -> C21 + A11 loc C22
        add(C, A, C, cx + new_size, cy, ax, ay, cx + new_size, cy + new_size, new_size);

        //18.U6 = U3 − P4 loc C21 -> C21 - A21 loc C21
        sub(C, A, C, cx + new_size, cy, ax + new_size, ay, cx + new_size, cy, new_size);

        //19.U4 = U2 + P5 loc C12 -> C12 + A11 loc C12
        add(C, A, C, cx, cy + new_size, ax, ay, cx, cy + new_size, new_size);

        //20. U5 = U4 + P3 loc C12 -> C12 + B11 loc C12
        add(C, B, C, cx, cy + new_size, bx, by, cx, cy + new_size, new_size);

        //21. P2 = A12 * B21 loc B11
        Mult_Strassen_Memory_efficient(A, B, B, ax, ay + new_size, bx + new_size, by, bx, by, new_size);

        //22. U1 = P1 + P2 loc C11 -> C11 + B11 loc C11
        add(C, B, C, cx, cy, bx, by, cx, cy, new_size);

    }

    friend Matrix Mult_Strassen_Memory_efficient(const Matrix &A, const Matrix &B) {
        Matrix C(A.size);
        Matrix Afake = A;
        Matrix Bfake = B;
        Mult_Strassen_Memory_efficient(Afake, Bfake, C, 0, 0, 0, 0, 0, 0, C.mx_size);

        return C;
    }

    friend void read_bin(const string &input, Matrix &A, Matrix &B) {
        A.clear();
        B.clear();
        ifstream in(input, ios::binary | ios::in);
        in.read((char *) &A.size, sizeof(A.size));
        A.mx_size = 1;
        while (A.mx_size < A.size) {
            A.mx_size *= 2;
        }
        vector<double> _m(A.size * A.size);
        in.read(reinterpret_cast<char *>(&_m[0]), A.size * A.size * sizeof(_m[0]));
        A.m.assign(A.mx_size * A.mx_size, 0);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                A.m[i * A.mx_size + j] = _m[i * A.size + j];
            }
        }
        in.read((char *) &B.size, sizeof(B.size));
        B.mx_size = A.mx_size;
        in.read(reinterpret_cast<char *>(&_m[0]), A.size * A.size * sizeof(_m[0]));
        B.m.assign(B.mx_size * B.mx_size, 0);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                B.m[i * B.mx_size + j] = _m[i * B.size + j];
            }
        }
    }

    friend void read_txt(const string &input, Matrix &A, Matrix &B) {
        A.clear();
        B.clear();
        ifstream in(input);
        in >> A >> B;
    }

    friend void output_bin(const string &output, Matrix &A) {
        ofstream out(output, ios::binary);
        out.write(reinterpret_cast<char *>(&A.size), sizeof(A.size));
        vector<double> _m(A.size * A.size);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                _m[i * A.size + j] = A.m[A.mx_size * i + j];
            }
        }
        out.write(reinterpret_cast<char *>(&_m[0]), A.size * A.size * sizeof(_m[0]));
    }

    friend ostream &operator<<(ostream &out, const Matrix &obj) {
        out << obj.size << '\n';
        for (int i = 0; i < obj.size; ++i) {
            for (int j = 0; j < obj.size; ++j) {
                out << obj.m[i * obj.mx_size + j] << ' ';
            }
            out << '\n';
        }
        return out;
    }

    friend istream &operator>>(istream &in, Matrix &obj) {
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


int main(int argc, char *argv[]) {

    const string input_file_bin = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.bin";
    const string input_file_txt = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.txt";
    const string output_time = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output.txt";
    const string my_ans = "C:/Users/Zver/Source/Repos/Project4/x64/Release/my_ans.bin";

    /*
    

    if (argc > 3) {
        input_file_bin = argv[1];
        output_time = argv[2];
        native_ans = argv[3];
    }
    */

    Matrix A, B, C;
    //54531948
    //46291219
    //52243276
    read_bin(input_file_bin, A, B);
    ofstream out(output_time);
    auto start = chrono::high_resolution_clock::now();
    //C = Mult_Strassen_Memory_efficient(A, B);
    //cout << C << '\n';
    //cout << "======\n";
    C = mult_block(A, B);
    //cout << C << '\n';
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = (end - start);
    duration *= 1000.0 * 1000.0;
    out.precision(10);
    out << fixed << duration.count() << '\n';
    output_bin(my_ans, C);
    return 0;
}