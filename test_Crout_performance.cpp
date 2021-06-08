// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <vector>
#include <limits>

#include "Eigen/Dense"
#include "Eigen/LU"

using namespace Eigen;
using namespace std;

#define DIM 8
#define LOOPS 50000


/**
 * \brief Crout matrix decomposition : in-place LU Decomposition of matrix A.
 *
 * LU decomposition function.
 * Implementation details : http://www.mymathlib.com/c_source/matrices/linearsystems/crout_pivot.c
 *
 * \param n The number of rows or columns of the matrix A
 * \param A matrix of size nxn : in-place LU decomposition (C-style arrays : row-major order)
 * \param pivot matrix of size n : The i-th element is the pivot row interchanged with row i
 */
#ifdef _OPENACC
#pragma acc routine seq
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void Crout(int n, T* A, int* pivot) {
    int i, j, k;
    T *p_k, *p_row, *p_col;
    T max;

    // For each row and column, k = 0, ..., n-1,
    for (k = 0, p_k = A; k < n; p_k += n, k++) {
        // find the pivot row
        pivot[k] = k;
        max = fabs(*(p_k + k));
        for (j = k + 1, p_row = p_k + n; j < n; j++, p_row += n) {
            if (max < fabs(*(p_row + k))) {
                max = fabs(*(p_row + k));
                pivot[k] = j;
                p_col = p_row;
            }
        }

        // and if the pivot row differs from the current row, then
        // interchange the two rows.
        if (pivot[k] != k)
            for (j = 0; j < n; j++) {
                max = *(p_k + j);
                *(p_k + j) = *(p_col + j);
                *(p_col + j) = max;
            }

        // and if the matrix is singular, return error
        // if ( *(p_k + k) == 0.0 ) return -1;

        // otherwise find the upper triangular matrix elements for row k.
        for (j = k + 1; j < n; j++) {
            *(p_k + j) /= *(p_k + k);
        }

        // update remaining matrix
        for (i = k + 1, p_row = p_k + n; i < n; p_row += n, i++)
            for (j = k + 1; j < n; j++)
                *(p_row + j) -= *(p_row + k) * *(p_k + j);
    }
    //return 0;
}

/**
 * \brief Crout matrix decomposition : Forward/Backward substitution.
 *
 * Forward/Backward substitution function.
 * Implementation details : http://www.mymathlib.com/c_source/matrices/linearsystems/crout_pivot.c
 *
 * \param n The number of rows or columns of the matrix LU
 * \param LU LU-factorized matrix (C-style arrays : row-major order)
 * \param B rhs vector
 * \param x solution of (LU)x=B linear system
 * \param pivot matrix of size n : The i-th element is the pivot row interchanged with row i
 */
#ifdef _OPENACC
#pragma acc routine seq
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void solveCrout(int n, T* LU, T* B, T* x, int* pivot) {
    int i, k;
    T* p_k;
    T dum;

    // Solve the linear equation Lx = B for x, where L is a lower
    // triangular matrix.
    for (k = 0, p_k = LU; k < n; p_k += n, k++) {
        if (pivot[k] != k) {
            dum = B[k];
            B[k] = B[pivot[k]];
            B[pivot[k]] = dum;
        }
        x[k] = B[k];
        for (i = 0; i < k; i++)
            x[k] -= x[i] * *(p_k + i);
        x[k] /= *(p_k + k);
    }

    // Solve the linear equation Ux = y, where y is the solution
    // obtained above of Lx = B and U is an upper triangular matrix.
    // The diagonal part of the upper triangular part of the matrix is
    // assumed to be 1.0.
    for (k = n - 1, p_k = LU + n * (n - 1); k >= 0; k--, p_k -= n) {
        if (pivot[k] != k) {
            dum = B[k];
            B[k] = B[pivot[k]];
            B[pivot[k]] = dum;
        }
        for (i = k + 1; i < n; i++)
            x[k] -= x[i] * *(p_k + i);
        // if (*(p_k + k) == 0.0) return -1;
    }

    //return 0;
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
    Matrix<int, DIM, 1> pivot[LOOPS];

    for (int i = 0; i < LOOPS; i++)
    {
        MatType A_;
        VecType b_, x_eigen_, x_crout_;

        do {
            // initialization
            for(int r = 0; r <  DIM; r++) {
                for(int c = 0; c < DIM; c++) {
                    A_(r,c) = nums(mt);
                    b_(r) = nums(mt);
                }
                x_eigen_(r) = (T)0;
                x_crout_(r) = (T)0;
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
    #pragma acc parallel loop copyin(A[0:LOOPS], b[0:LOOPS], pivot[0:LOOPS]) copyout(x_crout[0:LOOPS])
    for (int i = 0; i < LOOPS; i++)
    {
        Crout<T>(DIM, A[i].data(), pivot[i].data()); // in-place LU decomposition
        solveCrout<T>(DIM, A[i].data(), b[i].data(), x_crout[i].data(), pivot[i].data());
    }

    // Check Correctness
    for (int i = 0; i < LOOPS; i++)
       if (!allclose(x_eigen[i], x_crout[i], rtol, atol))
            return false;

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout_performance<double>(1e-8, 1e-8) << endl;

    return 0;
}