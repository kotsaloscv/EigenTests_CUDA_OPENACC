// nvcc -std=c++11 --expt-relaxed-constexpr -I./eigen inverse_cuda.cu -o inverse_cuda

#include <cmath>
#include <iostream>
#include <random>
#include "Eigen/LU"

#define DIM 4

using MatType = Eigen::Matrix<double, DIM, DIM>;

void inverse(MatType* I, MatType* J) {
    (*J) = (*I).inverse();
}

__global__ void cu_inverse(MatType* I, MatType* J) {
    (*J) = (*I).inverse();
}

int main() {
    MatType *I, *J, *K;

    cudaMallocManaged((void**)&I, DIM*DIM*sizeof(double));
    cudaMallocManaged((void**)&J, DIM*DIM*sizeof(double));
    cudaMallocManaged((void**)&K, DIM*DIM*sizeof(double));

    std::random_device rd; // seeding
    std::mt19937 mt(32); //std::mt19937 mt(rd());
    std::uniform_real_distribution<double> nums(-1.0, 1.0);
    for(int i = 0; i <  DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            (*I)(i,j) = nums(mt);
            (*J)(i,j) = 0;
            (*K)(i,j) = 0;
        }
    }
    std::cout << "MAT : " << (*I) << std::endl << std::endl;

    inverse(I, J);
    std::cout << "CPU INV " << std::endl << (*J) << std::endl << std::endl;

    cu_inverse<<<1,1>>>(I, K);
    cudaDeviceSynchronize();
    std::cout << "GPU INV " << std::endl << (*K) << std::endl << std::endl;

    return 0;
}