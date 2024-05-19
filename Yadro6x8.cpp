#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <map>
#include <string>
#include <omp.h>
#include <immintrin.h>

using namespace std;
constexpr double eps = 0.000001;

//loadu

inline void my_micro_yadro2(double* A, double* B, double* C, int offset, int small_z) {
    //A - 6 x zz
    //B - zz x 8
    //C - 6 x 8
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
        __m256d a1 = _mm256_set1_pd(*(A + i));
        __m256d a2 = _mm256_set1_pd(*(A + i + offset));

        __m256d b1 = _mm256_load_pd(B);
        __m256d b2 = _mm256_load_pd(B + 4);


        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c12 = _mm256_fmadd_pd(a1, b2, c12);

        c21 = _mm256_fmadd_pd(a2, b1, c21);
        c22 = _mm256_fmadd_pd(a2, b2, c22);

        a1 = _mm256_set1_pd(*(A + i + offset * 2));
        a2 = _mm256_set1_pd(*(A + i + offset * 3));

        c31 = _mm256_fmadd_pd(a1, b1, c31);
        c32 = _mm256_fmadd_pd(a1, b2, c32);

        c41 = _mm256_fmadd_pd(a2, b1, c41);
        c42 = _mm256_fmadd_pd(a2, b2, c42);

        a1 = _mm256_set1_pd(*(A + i + offset * 4));
        a2 = _mm256_set1_pd(*(A + i + offset * 5));

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

inline void my_micro_yadro3(double* A, double* B, double* C, int offset, int small_z) {
    //A - 8 x zz
    //B - zz x (8 * 3)
    //C - 8 x (8 * 3)
    
    register __m512d c11 asm("zmm0") = _mm512_setzero_pd();
    register __m512d c12 asm("zmm1") = _mm512_setzero_pd(); 
    register __m512d c13 asm("zmm2") = _mm512_setzero_pd();
    register __m512d c21 asm("zmm3") = _mm512_setzero_pd();
    register __m512d c22 asm("zmm4") = _mm512_setzero_pd();
    register __m512d c23 asm("zmm5") = _mm512_setzero_pd();
    register __m512d c31 asm("zmm6") = _mm512_setzero_pd();
    register __m512d c32 asm("zmm7") = _mm512_setzero_pd(); 
    register __m512d c33 asm("zmm8") = _mm512_setzero_pd(); 
    register __m512d c41 asm("zmm9") = _mm512_setzero_pd(); 
    register __m512d c42 asm("zmm10")= _mm512_setzero_pd();
    register __m512d c43 asm("zmm11")= _mm512_setzero_pd();
    register __m512d c51 asm("zmm12")= _mm512_setzero_pd();
    register __m512d c52 asm("zmm13")= _mm512_setzero_pd();
    register __m512d c53 asm("zmm14")= _mm512_setzero_pd();
    register __m512d c61 asm("zmm15")= _mm512_setzero_pd();
    register __m512d c62 asm("zmm16")= _mm512_setzero_pd();
    register __m512d c63 asm("zmm17")= _mm512_setzero_pd();
    register __m512d c71 asm("zmm18")= _mm512_setzero_pd();
    register __m512d c72 asm("zmm19")= _mm512_setzero_pd();
    register __m512d c73 asm("zmm20")= _mm512_setzero_pd();
    register __m512d c81 asm("zmm21")= _mm512_setzero_pd();
    register __m512d c82 asm("zmm22")= _mm512_setzero_pd();
    register __m512d c83 asm("zmm23")= _mm512_setzero_pd();
    
    register __m512d b11 asm("zmm24");
    register __m512d b12 asm("zmm25");
    register __m512d b13 asm("zmm26");
    register __m512d b21 asm("zmm27");
    register __m512d b22 asm("zmm28");
    register __m512d b23 asm("zmm29");
    register __m512d a1  asm("zmm30");
    register __m512d a2  asm("zmm31");
    for (int i = 0; i+1 < small_z; i+=2) {
        b11 = _mm512_loadu_pd(B);
        b12 = _mm512_loadu_pd(B + 8);
        b13 = _mm512_loadu_pd(B + 2 * 8);
        b21 = _mm512_loadu_pd(offset + B);
        b22 = _mm512_loadu_pd(offset + B + 8);
        b23 = _mm512_loadu_pd(offset + B + 2 * 8);
        a1 = _mm512_set1_pd(*(A + i));
        a2 = _mm512_set1_pd(*(A + i + offset));

        c11 = _mm512_fmadd_pd(a1, b11, c11);
        c12 = _mm512_fmadd_pd(a1, b12, c12);
        c13 = _mm512_fmadd_pd(a1, b13, c13);
        
        c21 = _mm512_fmadd_pd(a2, b11, c21);
        c22 = _mm512_fmadd_pd(a2, b12, c22);
        c23 = _mm512_fmadd_pd(a2, b13, c23);
        
        a1 = _mm512_set1_pd(*(A + i + offset * 2));
        a2 = _mm512_set1_pd(*(A + i + offset * 3));    
        
        c31 = _mm512_fmadd_pd(a1, b11, c31);
        c32 = _mm512_fmadd_pd(a1, b12, c32);
        c33 = _mm512_fmadd_pd(a1, b13, c33);
    
        c41 = _mm512_fmadd_pd(a2, b11, c41);
        c42 = _mm512_fmadd_pd(a2, b12, c42);
        c43 = _mm512_fmadd_pd(a2, b13, c43);
        
        a1 = _mm512_set1_pd(*(A + i + offset * 4));
        a2 = _mm512_set1_pd(*(A + i + offset * 5));    
        
        c51 = _mm512_fmadd_pd(a1, b11, c51);
        c52 = _mm512_fmadd_pd(a1, b12, c52);
        c53 = _mm512_fmadd_pd(a1, b13, c53);
    
        c61 = _mm512_fmadd_pd(a2, b11, c61);
        c62 = _mm512_fmadd_pd(a2, b12, c62);
        c63 = _mm512_fmadd_pd(a2, b13, c63);

        a1 = _mm512_set1_pd(*(A + i + offset * 6));
        a2 = _mm512_set1_pd(*(A + i + offset * 7));    
        
        c71 = _mm512_fmadd_pd(a1, b11, c71);
        c72 = _mm512_fmadd_pd(a1, b12, c72);
        c73 = _mm512_fmadd_pd(a1, b13, c73);
    
        c81 = _mm512_fmadd_pd(a2, b11, c81);
        c82 = _mm512_fmadd_pd(a2, b12, c82);
        c83 = _mm512_fmadd_pd(a2, b13, c83);
        
        //----
        a1 = _mm512_set1_pd(*(A + i + 1));
        a2 = _mm512_set1_pd(*(A + i + offset + 1));

        c11 = _mm512_fmadd_pd(a1, b21, c11);
        c12 = _mm512_fmadd_pd(a1, b22, c12);
        c13 = _mm512_fmadd_pd(a1, b23, c13);
        
        c21 = _mm512_fmadd_pd(a2, b21, c21);
        c22 = _mm512_fmadd_pd(a2, b22, c22);
        c23 = _mm512_fmadd_pd(a2, b23, c23);
        
        a1 = _mm512_set1_pd(*(A + i + offset * 2 + 1));
        a2 = _mm512_set1_pd(*(A + i + offset * 3 + 1));    
        
        c31 = _mm512_fmadd_pd(a1, b21, c31);
        c32 = _mm512_fmadd_pd(a1, b22, c32);
        c33 = _mm512_fmadd_pd(a1, b23, c33);
    
        c41 = _mm512_fmadd_pd(a2, b21, c41);
        c42 = _mm512_fmadd_pd(a2, b22, c42);
        c43 = _mm512_fmadd_pd(a2, b23, c43);
        
        a1 = _mm512_set1_pd(*(A + i + offset * 4 + 1));
        a2 = _mm512_set1_pd(*(A + i + offset * 5 + 1));    
        
        c51 = _mm512_fmadd_pd(a1, b21, c51);
        c52 = _mm512_fmadd_pd(a1, b22, c52);
        c53 = _mm512_fmadd_pd(a1, b23, c53);
    
        c61 = _mm512_fmadd_pd(a2, b21, c61);
        c62 = _mm512_fmadd_pd(a2, b22, c62);
        c63 = _mm512_fmadd_pd(a2, b23, c63);

        a1 = _mm512_set1_pd(*(A + i + offset * 6 + 1));
        a2 = _mm512_set1_pd(*(A + i + offset * 7 + 1));    
        
        c71 = _mm512_fmadd_pd(a1, b21, c71);
        c72 = _mm512_fmadd_pd(a1, b22, c72);
        c73 = _mm512_fmadd_pd(a1, b23, c73);
    
        c81 = _mm512_fmadd_pd(a2, b21, c81);
        c82 = _mm512_fmadd_pd(a2, b22, c82);
        c83 = _mm512_fmadd_pd(a2, b23, c83);

        B += 2 * offset;
    }
    
    a1  = _mm512_loadu_pd(C);//c11
    a2  = _mm512_loadu_pd(C + 8);//c12
    b11 = _mm512_loadu_pd(C + 2 * 8);//c13
    b12 = _mm512_loadu_pd(offset + C);//c21
    b13 = _mm512_loadu_pd(offset + C + 8);//c22
    b21 = _mm512_loadu_pd(offset + C + 2 * 8);//c23
    b22 = _mm512_loadu_pd(2 * offset + C);//c31
    b23 = _mm512_loadu_pd(2 * offset + C + 8);//c32

    a1 = _mm512_add_pd(a1, c11);
    a2 = _mm512_add_pd(a2, c12);
    b11 = _mm512_add_pd(b11,c13); 
    b12 = _mm512_add_pd(b12,c21);
    b13 = _mm512_add_pd(b13,c22);
    b21 = _mm512_add_pd(b21,c23);
    b22 = _mm512_add_pd(b22,c31);
    b23 = _mm512_add_pd(b23,c32);

    _mm512_storeu_pd(C        , a1);
    _mm512_storeu_pd(C + 1 * 8, a2);
    _mm512_storeu_pd(C + 2 * 8, b11);

    _mm512_storeu_pd(offset + C        , b12);
    _mm512_storeu_pd(offset + C + 1 * 8, b13);
    _mm512_storeu_pd(offset + C + 2 * 8, b21);
    
    _mm512_storeu_pd(2 * offset + C        , b22);
    _mm512_storeu_pd(2 * offset + C + 1 * 8, b23);
    

    a1  = _mm512_loadu_pd(2 * offset + C + 2 * 8);//c33 
    a2  = _mm512_loadu_pd(3 * offset + C        );//c41
    b11 = _mm512_loadu_pd(3 * offset + C + 8    );//c42
    b12 = _mm512_loadu_pd(3 * offset + C + 2 * 8);//c43
    b13 = _mm512_loadu_pd(4 * offset + C        );//c51
    b21 = _mm512_loadu_pd(4 * offset + C + 8    );//c52
    b22 = _mm512_loadu_pd(4 * offset + C + 2 * 8);//c53
    b23 = _mm512_loadu_pd(5 * offset + C        );//c61

    a1 = _mm512_add_pd(a1, c33);
    a2 = _mm512_add_pd(a2, c41);
    b11 = _mm512_add_pd(b11,c42); 
    b12 = _mm512_add_pd(b12,c43);
    b13 = _mm512_add_pd(b13,c51);
    b21 = _mm512_add_pd(b21,c52);
    b22 = _mm512_add_pd(b22,c53);
    b23 = _mm512_add_pd(b23,c61);


    _mm512_storeu_pd(2 * offset + C + 2 * 8, a1);
    _mm512_storeu_pd(3 * offset + C        , a2);
    _mm512_storeu_pd(3 * offset + C + 8    , b11);
    _mm512_storeu_pd(3 * offset + C + 2 * 8, b12);
    _mm512_storeu_pd(4 * offset + C        , b13);
    _mm512_storeu_pd(4 * offset + C + 8    , b21);
    _mm512_storeu_pd(4 * offset + C + 2 * 8, b22);
    _mm512_storeu_pd(5 * offset + C        , b23);


    a1  = _mm512_loadu_pd(5 * offset + C + 8    );//c62
    a2  = _mm512_loadu_pd(5 * offset + C + 2 * 8);//c63
    b11 = _mm512_loadu_pd(6 * offset + C        );//c71
    b12 = _mm512_loadu_pd(6 * offset + C + 8    );//c72
    b13 = _mm512_loadu_pd(6 * offset + C + 2 * 8);//c73
    b21 = _mm512_loadu_pd(7 * offset + C        );//c81
    b22 = _mm512_loadu_pd(7 * offset + C + 8    );//c82
    b23 = _mm512_loadu_pd(7 * offset + C + 2 * 8);//c83
    
    
    a1 = _mm512_add_pd(a1, c62);
    a2 = _mm512_add_pd(a2, c63);
    b11 = _mm512_add_pd(b11,c71); 
    b12 = _mm512_add_pd(b12,c72);
    b13 = _mm512_add_pd(b13,c73);
    b21 = _mm512_add_pd(b21,c81);
    b22 = _mm512_add_pd(b22,c82);
    b23 = _mm512_add_pd(b23,c83);

    _mm512_storeu_pd(5 * offset + C + 8    , a1);
    _mm512_storeu_pd(5 * offset + C + 2 * 8, a2);
    _mm512_storeu_pd(6 * offset + C        , b11);
    _mm512_storeu_pd(6 * offset + C + 8    , b12);
    _mm512_storeu_pd(6 * offset + C + 2 * 8, b13);
    _mm512_storeu_pd(7 * offset + C        , b21);
    _mm512_storeu_pd(7 * offset + C + 8    , b22);
    _mm512_storeu_pd(7 * offset + C + 2 * 8, b23);

}


class Matrix {
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
                    if (abs(m[i * mx_size + j] - B.m[i * mx_size + j]) > eps) {
                        cout << i << ' ' << j << '\n';
                        return false;
                    }
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

    
    friend Matrix mult_block_right(const Matrix& A, const Matrix& B) {

        int N = A.size;
        constexpr int m = 6 * 6 * 1;//x = 36
        constexpr int p = 8 * 20 * 1;//y = 160
        constexpr int n = 32 * 10 * 1;//z = 320

        constexpr int small_x = 6;
        constexpr int small_y = 8;
        constexpr int small_z = 64;


        Matrix ans(N);
        double sizekb = m * n + n * p + m * p;
        //sizekb = small_x * small_y + small_x * small_z + small_z * small_y;
        sizekb *= 8;
        sizekb /= 1024;
        int offset = A.mx_size;
        //cout << sizekb << '\n';
        
        #pragma omp parallel for proc_bind(close)
        for (int y = 0; y < N; y += p)
            for (int z = 0; z < N; z += n)
                for (int x = 0; x < N; x += m) {

                    int cnt_x = (min(N, x + m) - x) / small_x;
                    int cnt_y = (min(N, y + p) - y) / small_y;
                    int cnt_z = (min(N, z + n) - z) / small_z;
                    //    zyx
                    //0 - 000
                    for (int k = 0; k < cnt_z; k++)
                        for (int j = 0; j < cnt_y; j++)
                            for (int i = 0; i < cnt_x; i++) {
                                my_micro_yadro2(
                                    (double*)A.m.data() + (x + i * small_x) * A.mx_size + (z + k * small_z),
                                    (double*)B.m.data() + (z + k * small_z) * A.mx_size + (y + j * small_y),
                                    (double*)ans.m.data() + (x + i * small_x) * A.mx_size + (y + j * small_y),
                                    offset, small_z);

                            }
                                
                    for (int i = cnt_x * small_x + x; i < min(x + m, N); i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y; j < min(y + p, N); j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                            }
                        }
                    }
                    for (int i = x; i < cnt_x * small_x + x; i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y + small_y * cnt_y; j < min(y + p, N); j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                            }
                        }
                    }
                    for (int i = x; i <x + small_x * cnt_x; i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z + small_z * cnt_z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y; j < y + small_y * cnt_y; j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                            }
                        }
                    }
                }
        return ans;
    }

    friend Matrix mult_block(const Matrix& A, const Matrix& B) {

        int N = A.size;
        constexpr int m = 48;//x
        constexpr int p = 288;//y
        constexpr int n = 576;//z

        constexpr int small_x = 8;
        constexpr int small_y = 3 * 8;
        constexpr int small_z = 64;

        Matrix ans(N);
        double sizekb = m * n + n * p + m * p;
        //sizekb = small_x * small_y + small_x * small_z + small_z * small_y;
        sizekb *= 8;
        sizekb /= 1024;
        int offset = A.mx_size;
        cout << sizekb << '\n';

        #pragma omp parallel for proc_bind(close)
        for (int y = 0; y < N; y += p)
            for (int z = 0; z < N; z += n)
                for (int x = 0; x < N; x += m) 
                {

                    int cnt_x = (min(N, x + m) - x) / small_x;
                    int cnt_y = (min(N, y + p) - y) / small_y;
                    int cnt_z = (min(N, z + n) - z) / small_z;
                    //    zyx
                    //0 - 000
                    for (int k = 0; k < cnt_z; k++)
                        for (int j = 0; j < cnt_y; j++)
                            for (int i = 0; i < cnt_x; i++)
                            {
                                my_micro_yadro3(
                                    (double*)A.m.data() + (x + i * small_x) * A.mx_size + (z + k * small_z),
                                    (double*)B.m.data() + (z + k * small_z) * A.mx_size + (y + j * small_y),
                                    (double*)ans.m.data() + (x + i * small_x) * A.mx_size + (y + j * small_y),
                                    offset, small_z);

                            }
                                
                    for (int i = cnt_x * small_x + x; i < min(x + m, N); i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y; j < min(y + p, N); j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                            }
                        }
                    }
                    for (int i = x; i < cnt_x * small_x + x; i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y + small_y * cnt_y; j < min(y + p, N); j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                            }
                        }
                    }
                    for (int i = x; i <x + small_x * cnt_x; i++) {
                        int C_const = i * A.mx_size;
                        for (int k = z + small_z * cnt_z; k < min(z + n, N); k++) {
                            int A_const = C_const + k;
                            int B_const = k * A.mx_size;
                            #pragma omp simd
                            for (int j = y; j < y + small_y * cnt_y; j++) {
                                ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
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

    const string input_file_bin = "input.bin";
    const string input_file_txt = "input.txt";
    const string output_time = "output.txt";
    const string my_ans = "my_ans.bin";

    ofstream out(output_time,ios::binary);
    
    Matrix A, B, C, C2;
    read_bin(input_file_bin, A, B);
    auto start = chrono::high_resolution_clock::now();
    C = mult_block(A, B);
    auto end = chrono::high_resolution_clock::now();
    // C2 = mult_block_right(A, B);
    // if(C != C2){
    //     cout << "My mult is WRONG!\n";
    // }
    
    chrono::duration<double> duration = (end - start);
    duration *= 1000.0 * 1000.0;
    out.precision(10);
    cout << fixed << duration.count() << '\n';
    output_bin(my_ans, C);
    return 0;
}