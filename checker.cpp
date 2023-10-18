#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;
constexpr double eps = 0.00000001;

int main(int argc, char* argv[]) {


    const string my_bin_file = "C:/Users/Zver/Source/Repos/Project4/x64/Release/my_ans.bin";
    const string mkl_bin_file = "C:/Users/Zver/Source/Repos/Project4/x64/Release/output_mkl.bin";
    const string result = "C:/Users/Zver/Source/Repos/Project4/x64/Release/checker_verdict.txt";
    int n,n2;
    ifstream my_in(my_bin_file, ios::binary);
    ifstream mkl_in(mkl_bin_file, ios::binary);
    my_in.read((char*)&n, sizeof(int));
    double* A = new double[n * n];
    double* B = new double[n * n];
    my_in.read((char*)(A), sizeof(double) * n * n);
    
    mkl_in.read((char*)&n2, sizeof(int));
    mkl_in.read((char*)(B), sizeof(double) * n * n);
    int ok = 1;
    if (n != n2)
        ok = 0;
    cout.precision(10);
    cout << fixed;
    if (ok) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (abs(A[i * n + j] - B[i * n + j]) > eps) {
                    ok = 0;
                    break;
                }
            }
        }
    }
    ofstream out(result);
    if (ok) {
        out << "Correct!\n";
    }
    else
        out << "Not Correct!\n";
    return 0;

}