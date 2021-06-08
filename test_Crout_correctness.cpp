// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <limits>

#include "Eigen/Dense"
#include "Eigen/LU"

using namespace Eigen;
using namespace std;


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
bool test_Crout_correctness(T rtol = 1e-6, T atol = 1e-6)
{
    using MatType = Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>;
    using VecType = Matrix<T, Dynamic, 1>;

    std::random_device rd; // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-10, 10);
    
    std::chrono::duration<double> eigen_solve_RowMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> eigen_solve_ColMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> crout_solve_host(std::chrono::duration<double>::zero());

    T max_relative_error_eigen = std::numeric_limits<T>::epsilon();
    T max_relative_error_crout = std::numeric_limits<T>::epsilon();

    for (int mat_size = 2; mat_size < 10; mat_size++)
    {
        MatType A_RowMajor(mat_size, mat_size);
        Matrix<T, Dynamic, Dynamic, Eigen::ColMajor> A_ColMajor(mat_size, mat_size); // default in Eigen!
        VecType b(mat_size);
        
        for (int repetitions = 0; repetitions < 100000; ++repetitions)
        {
            do
            {
                // initialization
                for(int r = 0; r < mat_size; r++) {
                    for(int c = 0; c < mat_size; c++) {
                        A_RowMajor(r,c) = nums(mt);
                        A_ColMajor(r,c) = A_RowMajor(r,c);
                        b(r) = nums(mt);
                    }
                }
            } while (!A_RowMajor.fullPivLu().isInvertible()); // Checking Invertibility
            
            // Eigen (RowMajor)
            VecType eigen_solution_RowMajor(mat_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_RowMajor = A_RowMajor.partialPivLu().solve(b);
            auto t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_RowMajor += (t2 - t1);
            T relative_error_eigen = (A_RowMajor*eigen_solution_RowMajor - b).norm() / b.norm(); // norm() is L2 norm
            if (relative_error_eigen > max_relative_error_eigen)
                max_relative_error_eigen = relative_error_eigen;

            // Eigen (ColMajor)
            VecType eigen_solution_ColMajor(mat_size);
            t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_ColMajor = A_ColMajor.partialPivLu().solve(b);
            t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_ColMajor += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, eigen_solution_ColMajor, rtol, atol)) {
                cerr << "Eigen issue with RowMajor vs ColMajor storage order!" << endl << endl;
                return false;
            }

            // Crout LU-Decomposition CPU
            MatType LU(mat_size, mat_size);
            LU = A_RowMajor;

            Matrix<int, Dynamic, 1> pivot(mat_size);
            VecType crout_solution_host(mat_size);
            
            t1 = std::chrono::high_resolution_clock::now();
            Crout<T>(mat_size, LU.data(), pivot.data());
            solveCrout<T>(mat_size, LU.data(), b.data(), crout_solution_host.data(), pivot.data());
            t2 = std::chrono::high_resolution_clock::now();
            crout_solve_host += (t2 - t1);
            T relative_error_crout = (A_RowMajor*crout_solution_host - b).norm() / b.norm(); // norm() is L2 norm
            if (relative_error_crout > max_relative_error_crout)
                max_relative_error_crout = relative_error_crout;

            if (!allclose(eigen_solution_RowMajor, crout_solution_host, rtol, atol)) {
                return false;
            }

#ifdef GPU
            // Crout LU-Decomposition GPU
            VecType crout_solution_dev(mat_size);
            LU = A_RowMajor;

            T *LU_dev = LU.data();
            T *b_dev  = b.data();
            T *x_dev  = crout_solution_dev.data();
            int *pivot_dev = pivot.data();

            #pragma acc kernels copyin(LU_dev[0:mat_size*mat_size], b_dev[0:mat_size], pivot_dev[0:mat_size]) copyout(x_dev[0:mat_size])
            {

                Crout<T>(mat_size, LU_dev, pivot_dev);
                solveCrout<T>(mat_size, LU_dev, b_dev, x_dev, pivot_dev);
            }

            if (!allclose(eigen_solution_RowMajor, crout_solution_dev, rtol, atol)) {
                return false;
            }
#endif
        }
    }

    cout << "Eigen RowMajor : " << eigen_solve_RowMajor.count()*1e3 << " ms" << endl;
    cout << "Eigen ColMajor : " << eigen_solve_ColMajor.count()*1e3 << " ms" << endl;
    cout << "Crout host     : " << crout_solve_host.count()*1e3 << " ms" << endl;

    cout << "Eigen relative error : " << max_relative_error_eigen << endl;
    cout << "Crout relative error : " << max_relative_error_crout << endl;

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout_correctness<double>(1e-8, 1e-8) << endl;

    return 0;
}