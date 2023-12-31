#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include <omp.h>
#include "mkl.h"

using namespace std;


int main(int argc, char* argv[]) {

    const string input_file_bin = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.bin";
    const string input_file_txt = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.txt";
    const string output_time = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output_mkl_time.txt";
    const string ans = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output_mkl.bin";

    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    int n = 3;

    ifstream in(input_file_bin, ios::binary);
    in.read((char*)&n, sizeof(int));
    
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C = new double[n * n];

    in.read((char*)(A), sizeof(double) * n * n);
    in.read((char*)&n, sizeof(int));
    in.read((char*)(B), sizeof(double) * n * n);

    auto start = chrono::high_resolution_clock::now();
    
    ofstream out(output_time);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, alpha, A, n, B, n, beta, C, n);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = (end - start);
    duration *= 1000.0 * 1000.0;
    out.precision(10);
    out << fixed << duration.count() << '\n';
    ofstream fout(ans, ios::binary);
    fout.write((char*)&n, sizeof(int));
    fout.write((char*)(C), sizeof(double) * n * n);
    fout.close();

    return 0;
   
}