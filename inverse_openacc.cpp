// nvc++ -acc -DEIGEN_DONT_VECTORIZE=1 -I./eigen inverse_openacc.cpp -o inverse_openacc

#include <cmath>
#include <iostream>
#include <random>
#include "Eigen/LU"

#define DIM 4

using MatType = Eigen::Matrix<double, DIM, DIM>;

int main() {
    MatType I, J;

    std::random_device rd; // seeding
    std::mt19937 mt(32); //std::mt19937 mt(rd());
    std::uniform_real_distribution<double> nums(-1.0, 1.0);
    for(int i = 0; i <  DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            I(i,j) = nums(mt);
            J(i,j) = 0;
        }
    }
    std::cout << "MAT : " << I << std::endl << std::endl;

    J = I.inverse();
    std::cout << "CPU INV " << std::endl << J << std::endl << std::endl;

    double* I_ = I.data();
    double* K_ = new double[DIM*DIM];
    #pragma acc kernels copy(I_[0:DIM*DIM],K_[0:DIM*DIM])
    {
        MatType tmp, inv;

        for(int i = 0; i <  DIM; i++) {
            for(int j = 0; j < DIM; j++) {
                tmp(i,j) = I_[i*DIM + j];
            }
        }

        inv = tmp.inverse();
        //inv = tmp.partialPivLu().inverse();

        for(int i = 0; i <  DIM; i++) {
            for(int j = 0; j < DIM; j++) {
                K_[i*DIM + j] = inv(i,j);
            }
        }
    }
    std::cout << "OPENACC INV " << std::endl << Eigen::Map<MatType>(K_) << std::endl << std::endl;

    return 0;
}