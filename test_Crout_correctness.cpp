// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <random>
#include <chrono>

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
static inline void Crout(int d,T*S,T*D){
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
static inline void solveCrout(int d,T*LU,T*b,T*x){
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
bool test_Crout_correctness(T rtol = 1e-6, T atol = 1e-6) 
{
    using MatType = Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>;
    using VecType = Matrix<T, Dynamic, 1>;

    std::random_device rd; // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1, 1);
    
    std::chrono::duration<double> eigen_solve_RowMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> eigen_solve_ColMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> crout_solve_host(std::chrono::duration<double>::zero());

    for (int mat_size = 2; mat_size < 10; mat_size++)
    {
        MatType A_RowMajor(mat_size, mat_size);
        Matrix<T, Dynamic, Dynamic, Eigen::ColMajor> A_ColMajor(mat_size, mat_size); // default in Eigen!
        VecType b(mat_size);
        
        for (int repetitions = 0; repetitions < 1000000; ++repetitions)
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
            VecType crout_solution_host(mat_size);
            
            t1 = std::chrono::high_resolution_clock::now();
            Crout<T>(mat_size, A_RowMajor.data(), LU.data());
            solveCrout<T>(mat_size, LU.data(), b.data(), crout_solution_host.data());
            t2 = std::chrono::high_resolution_clock::now();
            crout_solve_host += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, crout_solution_host, rtol, atol)) {
                return false;
            }

#ifdef GPU
            // Crout LU-Decomposition GPU
            VecType crout_solution_dev(mat_size);

            T *A_dev  = A_RowMajor.data();
            T *LU_dev = LU.data();
            T *b_dev  = b.data();
            T *x_dev  = crout_solution_dev.data();

            #pragma acc kernels copyin(A_dev[0:mat_size*mat_size], LU_dev[0:mat_size*mat_size], b_dev[0:mat_size]) copyout(x_dev[0:mat_size])
            {
                Crout<T>(mat_size, A_dev, LU_dev);
                solveCrout<T>(mat_size, LU_dev, b_dev, x_dev);
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

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout_correctness<double>(1e-6, 1e-6) << endl;

    return 0;
}