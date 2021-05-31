// nvc++ -acc -DEIGEN_DONT_VECTORIZE=1 -I./eigen exp.cpp -o exp

#include <cmath>
#include <iostream>
#include <random>

#include "Eigen/LU"

using namespace Eigen;
using namespace std;

using T = double;


// Crout uses unit diagonals for the upper triangle
template<typename T_>
void Crout(int d,T_*S,T_*D){
   for(int k=0;k<d;++k){
      for(int i=k;i<d;++i){
         T_ sum=0.;
         for(int p=0;p<k;++p)sum+=D[i*d+p]*D[p*d+k];
         D[i*d+k]=S[i*d+k]-sum; // not dividing by diagonals
      }
      for(int j=k+1;j<d;++j){
         T_ sum=0.;
         for(int p=0;p<k;++p)sum+=D[k*d+p]*D[p*d+j];
         D[k*d+j]=(S[k*d+j]-sum)/D[k*d+k];
      }
   }
}
template<typename T_>
void solveCrout(int d,T_*LU,T_*b,T_*x){
   T_ y[d];
   for(int i=0;i<d;++i){
      T_ sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum)/LU[i*d+i];
   }
   for(int i=d-1;i>=0;--i){
      T_ sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[i*d+k]*x[k];
      x[i]=(y[i]-sum); // not dividing by diagonals
   }
}


int main(int argc, char** argv) {

    const int size = std::atoi(argv[1]);

    Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> A(size, size);
    Matrix<T, Dynamic, 1> b(size);

    std::random_device rd; // seeding
    std::mt19937 mt(32); //std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1.0, 1.0);
    
    for(int i = 0; i <  size; i++) {
        for(int j = 0; j < size; j++) {
            A(i,j) = nums(mt);
            b(i)   = nums(mt);
        }
    }
    cout << "A" << endl;
    cout << A << endl << endl;
    cout << "b" << endl;
    cout << b.transpose() << endl << endl;

    Matrix<T, Dynamic, 1> eigen_solution(size);
    eigen_solution = A.partialPivLu().solve(b);
    cout << "eigen solution CPU" << endl;
    cout << eigen_solution.transpose() << endl << endl;


    Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> LU(size, size);
    Matrix<T, Dynamic, 1> croot_solution(size);  

    Crout<T>(size, A.data(), LU.data());
    solveCrout<T>(size, LU.data(), b.data(), croot_solution.data());

    cout << "croot solution CPU" << endl;
    cout << croot_solution.transpose() << endl << endl;

    return 0;
}