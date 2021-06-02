// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/LU"

using namespace Eigen;
using namespace std;

#define DIM 4
#define LOOPS 50000

/**
 * \brief Crout matrix decomposition : LU Decomposition of (S)ource matrix stored in (D)estination matrix.
 * 
 * LU decomposition function.
 * Implementation details : http://www.sci.utah.edu/~wallstedt/LU.htm
 * 
 * \param d matrices of size d x d
 * \param S Source matrix (C-style arrays : row-major order)
 * \param D Destination matrix (LU decomposition of S-matrix) (C-style arrays : row-major order)
 */
#pragma acc routine seq
template<typename T>
void Crout(int d,T*S,T*D){
   for(int k=0;k<d;++k){
      for(int i=k;i<d;++i){
         T sum=0.;
         for(int p=0;p<k;++p)sum+=D[i*d+p]*D[p*d+k];
         D[i*d+k]=S[i*d+k]-sum;
      }
      for(int j=k+1;j<d;++j){
         T sum=0.;
         for(int p=0;p<k;++p)sum+=D[k*d+p]*D[p*d+j];
         D[k*d+j]=(S[k*d+j]-sum)/D[k*d+k];
      }
   }
}

/**
 * \brief Crout matrix decomposition : Forward/ Backward substitution.
 * 
 * Forward/ Backward substitution function.
 * Implementation details : http://www.sci.utah.edu/~wallstedt/LU.htm
 * 
 * \param d matrices of size d x d
 * \param LU LU-factorized matrix (C-style arrays : row-major order)
 * \param b rhs vector
 * \param x solution of (LU)x=b linear system
 */
#pragma acc routine seq
template<typename T>
void solveCrout(int d,T*LU,T*b,T*x){
   T y[d];
   for(int i=0;i<d;++i){
      T sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum)/LU[i*d+i];
   }
   for(int i=d-1;i>=0;--i){
      T sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[i*d+k]*x[k];
      x[i]=(y[i]-sum);
   }
}


/// https://stackoverflow.com/questions/15051367/how-to-compare-vectors-approximately-in-eigen
template<typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,
              const Eigen::DenseBase<DerivedB>& b,
              const typename DerivedA::RealScalar& rtol
                  = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar& atol
                  = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
{
  return ((a.derived() - b.derived()).array().abs()
          <= (atol + rtol * b.derived().array().abs())).all();
}


template<typename T>
bool test_Crout_performance(T rtol = 1e-6, T atol = 1e-6) 
{
    using MatType = Matrix<T, DIM, DIM, Eigen::RowMajor>;
    using VecType = Matrix<T, DIM, 1>;

    std::random_device rd; // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1, 1);
    
    MatType A[LOOPS];
    VecType b[LOOPS], x_eigen[LOOPS], x_crout[LOOPS];

    for (int i = 0; i < LOOPS; i++)
    {
        MatType A_;
        VecType b_, x_eigen_, x_crout_;
        do {
            for(int i = 0; i <  DIM; i++) {
                for(int j = 0; j < DIM; j++) {
                    A_(i,j) = nums(mt);
                    b_(i) = nums(mt);
                    
                    x_eigen_(i) = (T)0;
                    x_crout_(i) = (T)0;
                }
            }
        }
        while (!A_.fullPivLu().isInvertible()); // Checking Invertibility

        A[i] = A_;
        b[i] = b_;

        x_eigen[i] = x_eigen_;
        x_crout[i] = x_crout_;
    }

    // Eigen CPU
    std::chrono::duration<double> eigen_solve(std::chrono::duration<double>::zero());
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < LOOPS; i++)
    {
        x_eigen[i] = A[i].partialPivLu().solve(b[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    eigen_solve = (t2 - t1);
    cout << "Eigen : " << eigen_solve.count()*1e3 << " ms" << endl;

    // Crout
    #pragma acc parallel loop copyin(A[0:LOOPS], b[0:LOOPS]) copy(x_crout[0:LOOPS])
    for (int i = 0; i < LOOPS; i++)
    {
        Crout<T>(DIM, A[i].data(), A[i].data()); // in-place LU decomposition
        solveCrout<T>(DIM, A[i].data(), b[i].data(), x_crout[i].data());
    }

    // Check Correctness
    for (int i = 0; i < LOOPS; i++)
       if (!allclose(x_eigen[i], x_crout[i], rtol, atol))
            return false;

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout_performance<double>(1e-6, 1e-6) << endl;

    return 0;
}