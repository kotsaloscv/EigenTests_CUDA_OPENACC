#include <iostream>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <chrono>
 
using namespace Eigen;
using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(int argc, char** argv) {

    int size = std::atoi(argv[1]);
    int times = std::atoi(argv[2]);

    // Initialize random number generator with seed=42 for following codes.
    // Or you can use C++11 RNG such as std::mt19937 or std::ranlux48.
    Rand::Vmt19937_64 urng{ 42 };

    // this will generate 4x4 real matrix with range [-1, 1]
    MatrixXf A = Rand::balanced<MatrixXf>(size, size, urng);
    //std::cout << A << std::endl << std::endl;

    MatrixXf b = Rand::balanced<MatrixXf>(size, 1, urng);
    //std::cout << b.transpose() << std::endl << std::endl;


    {
        cout << "A.inverse()*b : The solution is:\n" << (A.inverse()*b).transpose() << "\n";
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < times; ++i) {
            A.inverse()*b;
        }
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        cout << ms_int.count() << "ms\n";
    }
    cout << endl;

    {
        cout << "A.partialPivLu().solve(b) : The solution is:\n" << (A.partialPivLu().solve(b)).transpose() << "\n";
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < times; ++i) {
            A.partialPivLu().solve(b);
        }
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        cout << ms_int.count() << "ms\n";
    }
    cout << endl;

    {
        cout << "A.fullPivLu().solve(b) : The solution is:\n" << (A.fullPivLu().solve(b)).transpose() << "\n";
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < times; ++i) {
            A.fullPivLu().solve(b);
        }
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        cout << ms_int.count() << "ms\n";
    }
    cout << endl;

    return 0;
}