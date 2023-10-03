
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <string>

#include <omp.h>
using namespace std;
constexpr double eps = 0.00000000001;

class Matrix {
    /*
    mx_size - размер , который является степенью двойки
    нужен для того, чтобы нормально работало перемножение матриц
    */
    int size;
    int mx_size;
    vector<double>m;
public:
    Matrix(int _size = 0) {
        size = _size;
        mx_size = 1;
        while (mx_size < size) {
            mx_size *= 2;
        }
        m.assign(mx_size * mx_size, 0);
    }

    Matrix(int _size , vector<double>& _m) {
        size = _size;
        mx_size = 1;
        while (mx_size < size) {
            mx_size *= 2;
        }
        m.assign(mx_size * mx_size, 0);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                m[i * mx_size + j] = _m[i*size + j];
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

    double& operator() (const int i, const int j) {
        return m[i * size + j];
    }

    Matrix operator + (const Matrix& B) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] + B.m[i * mx_size + j];
            }
        }
        return ans;
    }
    Matrix operator - (const Matrix& B) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] - B.m[i * mx_size + j];
            }
        }
        return ans;
    }

    Matrix operator *(double value) {
        Matrix ans(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                ans.m[i * mx_size + j] = m[i * mx_size + j] * value;
            }
        }
        return ans;
    }
    bool operator == (const Matrix& B) {
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

    bool operator != (const Matrix& B) {
        return !((*this) != B);
    }


    Matrix& operator =  (const Matrix& B) {
        if (&B == this) return *this;
        this->clear();
        size = B.size;
        mx_size = B.mx_size;
        m = B.m;
        return *this;
    }

    friend Matrix mult(Matrix& A , Matrix& B) {
        Matrix ans(A.size);
        //#pragma omp parallel for
        for (int i = 0; i < A.size; ++i) {
            int C_const = i * A.mx_size;
            for (int k = 0; k < A.size; ++k) {
                int A_const = C_const + k;
                int B_const = k * A.mx_size;
                #pragma omp simd
                for (int j = 0; j < A.size; ++j) {
                    ans.m[C_const + j] += A.m[A_const] * B.m[B_const + j];
                }
            }
        }
        return ans;
    }
    friend Matrix mult_block(Matrix& A, Matrix& B, int row_A, int col_A,int row_B , int col_B, int size) {

        /*
            (row,col) - индекс левой верхней клетки матрицы
        */
        
        if (size <= 8) {
            //Наивное умножение
            Matrix C(size);
            for (int i = 0; i < size; i++) {
                for (int k = 0; k < size; k++) {
                    for (int j = 0; j < size; j++) {
                        C.m[i * C.mx_size + j] += A.m[(i + row_A) * A.mx_size + (k + col_A)] * B.m[(k + row_B) * B.mx_size + (j + col_B)];
                    }
                }
            }
            return C;
        }
        int new_size = size / 2;
        Matrix C11 = mult_block(A, B, row_A, col_A, row_B, col_B, new_size) + mult_block(A, B, row_A, col_A + new_size,row_B + new_size,col_B,new_size);
        Matrix C12 = mult_block(A, B, row_A, col_A, row_B, col_B + new_size, new_size) + mult_block(A, B, row_A, col_A + new_size,row_B + new_size,col_B+new_size,new_size);
        Matrix C21 = mult_block(A, B, row_A+new_size, col_A, row_B, col_B, new_size) + mult_block(A, B, row_A+new_size, col_A + new_size,row_B+new_size,col_B,new_size);
        Matrix C22 = mult_block(A, B, row_A+new_size, col_A, row_B, col_B+new_size, new_size) + mult_block(A, B, row_A+new_size, col_A + new_size,row_B + new_size,col_B+new_size,new_size);
        Matrix C(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double& val = C.m[i * C.mx_size + j];
                if (i < new_size && j < new_size) {
                    val = C11.m[i * C11.mx_size + j];
                }
                else if (i < new_size && j >= new_size) {
                    val = C12.m[i * C12.mx_size + (j-new_size)];
                }
                else if(i >= new_size && j < new_size) {
                    val = C21.m[(i-new_size) * C21.mx_size + (j)];
                }
                else {
                    val = C22.m[(i - new_size) * C22.mx_size + (j-new_size)];
                }
            }
        }
        return C;
    }

    friend Matrix mult_Strassen(Matrix& A, Matrix& B, int size) {

        if (size <= 8) {
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
                }
                else if (i < new_size && j >= new_size) {
                    A12.m[i * new_size + (j-new_size)] = A.m[i * size + j];
                    B12.m[i * new_size + (j-new_size)] = B.m[i * size + j];
                }
                else if (i >= new_size && j < new_size) {
                    A21.m[(i-new_size) * new_size + j] = A.m[i * size + j];
                    B21.m[(i-new_size) * new_size + j] = B.m[i * size + j];
                }
                else {
                    A22.m[(i-new_size) * new_size + (j - new_size)] = A.m[i * size + j];
                    B22.m[(i-new_size) * new_size + (j - new_size)] = B.m[i * size + j];
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
        
        Matrix P1 = mult_Strassen(S2, S6,new_size);
        Matrix P2 = mult_Strassen(A11, B11,new_size);
        Matrix P3 = mult_Strassen(A12, B21,new_size);
        Matrix P4 = mult_Strassen(S3, S7,new_size);
        Matrix P5 = mult_Strassen(S1, S5,new_size);
        Matrix P6 = mult_Strassen(S4, B22,new_size);
        Matrix P7 = mult_Strassen(A22, S8,new_size);

        Matrix T1 = P1 + P2;
        Matrix T2 = T1 + P4;

        Matrix C11 = P2 + P3;
        Matrix C12 = T1 + P5 + P6;
        Matrix C21 = T2 - P7;
        Matrix C22 = T2 + P5;

        Matrix C(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double& val = C.m[i * C.mx_size + j];
                if (i < new_size && j < new_size) {
                    val = C11.m[i * C11.mx_size + j];
                }
                else if (i < new_size && j >= new_size) {
                    val = C12.m[i * C12.mx_size + (j - new_size)];
                }
                else if (i >= new_size && j < new_size) {
                    val = C21.m[(i - new_size) * C21.mx_size + (j)];
                }
                else {
                    val = C22.m[(i - new_size) * C22.mx_size + (j - new_size)];
                }
            }
        }
        return C;
    }

    friend Matrix mult_Strassen(Matrix& A, Matrix& B) {
        Matrix C = mult_Strassen(A, B, A.mx_size);

        C.size = A.size;
        return C;
    }

    friend Matrix mult_block(Matrix& A, Matrix& B) {
        Matrix C = mult_block(A, B, 0, 0, 0, 0, A.mx_size);
        C.size = A.size;
        return C;
    }

    friend void read_bin(const string& input, Matrix& A, Matrix& B) {
        A.clear();
        B.clear();
        ifstream in(input, ios::binary | ios::in);
        in.read((char*)&A.size, sizeof(A.size));
        A.mx_size = 1;
        while (A.mx_size < A.size) {
            A.mx_size *= 2;
        }
        vector<double>_m(A.size * A.size);
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
        vector<double>_m(A.size * A.size);
        for (int i = 0; i < A.size; i++) {
            for (int j = 0; j < A.size; j++) {
                _m[i * A.size + j] = A.m[A.mx_size * i + j];
            }
        }
        out.write(reinterpret_cast<char*>(&_m[0]), A.size * A.size * sizeof(_m[0]));       
    }

    friend ostream& operator << (ostream& out, const Matrix& obj) {
        out << obj.size << '\n';
        for (int i = 0; i < obj.size; ++i) {
            for (int j = 0; j < obj.size; ++j) {
                out << obj.m[i * obj.mx_size + j] << ' ';
            }
            out << '\n';
        }
        return out;
    }

    friend istream& operator >> (istream& in, Matrix& obj) {
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

    /*
        через командную строку передаем:
        1)Бинарник с входными данными
        2)Куда записать время работы программы
        3)Куда записать результат умножения
        string input_file_txt = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.txt";
    */
    
    const string input_file_bin = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.bin";
    const string output_time = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output.txt";
    const string native_ans = "C:/Users/Zver/Source/Repos/Project4/x64/Release/native_ans.bin";

    /*
    if (argc > 3) {
        input_file_bin = argv[1];
        output_time = argv[2];
        native_ans = argv[3];
    }
    */

    Matrix A,B,C,C2;
    
    read_bin(input_file_bin, A, B);
    ofstream out(output_time);
    auto start = chrono::high_resolution_clock::now();
    C = mult_Strassen(A,B);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = (end - start);
    duration *= 1000.0 * 1000.0;
    out.precision(10);
    out << fixed << duration.count() << '\n';
    output_bin(native_ans, C);
    return 0;
}