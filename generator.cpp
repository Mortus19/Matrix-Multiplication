#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <string>

using namespace std;
constexpr int INF = 1000;
mt19937 random_generator(chrono::steady_clock::now().time_since_epoch().count());

void gen(double* a, int n, int maxv) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i*n + j] = (int)(random_generator() % maxv - maxv / 2);
            a[i * n + j] /= 10.0;
        }
    }
}

void write_txt(string& output , int n , double* a , double* b) {
    ofstream out;
    out.open(output,'r');
    out.precision(3);

    out << n << '\n';
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << a[i * n + j] << ' ';
        }
        out << '\n';
    }
    out << n << '\n';
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << b[i * n + j] << ' ';
        }
        out << '\n';
    }
    out << '\n';
}

void write_bin(string& output, int n, double* a, double* b) {
    ofstream out;
    out.open(output,ios::binary);
    out.write((char*)&n, sizeof(int));
    out.write((char*)a, sizeof(double)*n*n);
    out.write((char*)&n, sizeof(int));
    out.write((char*)b, sizeof(double)*n*n);
}

int main(int argc, char* argv[]) {
    
    string output_txt = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.txt";
    string output_bin = "C:/Users/Zver/Source/Repos/Project4/x64/Release/input.bin";
    int n = 500;
    if (argc > 1) {
        string t = argv[1];
        n = stoi(t);
    }
    double* a = new double [n*n];
    double* b = new double [n*n];
    
    gen(a, n, random_generator() % INF);
    gen(b, n, random_generator() % INF);
    write_txt(output_txt, n, a, b);
    write_bin(output_bin, n, a, b);
    
    return 0;
}