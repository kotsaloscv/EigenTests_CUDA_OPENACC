// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include <omp.h>

#include "Eigen/Dense"
#include "Eigen/LU"

using namespace Eigen;
using namespace std;


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
    std::vector<Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>> A_;
    std::vector<Matrix<T, Dynamic, 1>> b_;
    std::vector<Matrix<T, Dynamic, 1>> x_eigen_;
    std::vector<Matrix<T, Dynamic, 1>> x_crout_;

    std::random_device rd; // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1, 1);
    
    for (int mat_size = 2; mat_size <= 10; mat_size++)
    {
        Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> A(mat_size, mat_size);
        Matrix<T, Dynamic, 1> b(mat_size);
        
        for (int i = 0; i < 100000; ++i) 
        {
            // initialization
            for(int i = 0; i <  mat_size; i++) {
                for(int j = 0; j < mat_size; j++) {
                    A(i,j) = nums(mt);
                    b(i) = nums(mt);
                }
            }

            // Checking Invertibility
            if (!A.fullPivLu().isInvertible())
                continue;

            A_.push_back(A);
            b_.push_back(b);
            x_eigen_.push_back(b); // just an initialization
            x_crout_.push_back(b); // just an initialization
        }
    }

    // Eigen CPU
    std::chrono::duration<double> eigen_solve(std::chrono::duration<double>::zero());
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < A_.size(); i++)
    {
        x_eigen_[i] = A_[i].partialPivLu().solve(b_[i]);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    eigen_solve = (t2 - t1);
    cout << "Eigen : " << eigen_solve.count()*1e3 << " ms" << endl;

    // Crout
    std::chrono::duration<double> crout_solve(std::chrono::duration<double>::zero());
    t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < A_.size(); i++)
    {
        int mat_size = b_[i].size();
        Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> LU(mat_size, mat_size);
        Crout<T>(mat_size, A_[i].data(), LU.data());
        solveCrout<T>(mat_size, LU.data(), b_[i].data(), x_crout_[i].data());
    }
    t2 = std::chrono::high_resolution_clock::now();
    crout_solve = (t2 - t1);
    cout << "Crout : " << crout_solve.count()*1e3 << " ms" << endl;

    // Check correctness
    for (int i = 0; i < A_.size(); i++) {
        if (!allclose(x_eigen_[i], x_crout_[i], rtol, atol))
            return false;
    }

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout_performance<double>(1e-6, 1e-6) << endl;

    return 0;
}