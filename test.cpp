// Legacy code : coreneuron/sim/scopmath/crout_thread.cpp

#include <iostream>
#include <random>
#include <chrono>

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
bool test_Crout(T rtol = 1e-6, T atol = 1e-6) 
{
    std::random_device rd; // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1, 1);
    
    std::chrono::duration<double> eigen_solve_RowMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> eigen_solve_ColMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> crout_solve(std::chrono::duration<double>::zero());

    for (int mat_size = 1; mat_size < 50; mat_size++)
    {
        Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> A_RowMajor(mat_size, mat_size);
        Matrix<T, Dynamic, Dynamic, Eigen::ColMajor> A_ColMajor(mat_size, mat_size); // default in Eigen!
        Matrix<T, Dynamic, 1> b(mat_size);
        
        for (int i = 0; i < 10000; ++i) 
        {
            // initialization
            for(int i = 0; i <  mat_size; i++) {
                for(int j = 0; j < mat_size; j++) {
                    A_RowMajor(i,j) = nums(mt);
                    A_ColMajor(i,j) = A_RowMajor(i,j);
                    b(i) = nums(mt);
                }
            }

            // Eigen (RowMajor)
            Matrix<T, Dynamic, 1> eigen_solution_RowMajor(mat_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_RowMajor = A_RowMajor.partialPivLu().solve(b);
            auto t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_RowMajor += (t2 - t1);

            // Eigen (ColMajor)
            Matrix<T, Dynamic, 1> eigen_solution_ColMajor(mat_size);
            t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_ColMajor = A_ColMajor.partialPivLu().solve(b);
            t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_ColMajor += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, eigen_solution_ColMajor, rtol, atol)) {
                cerr << "Eigen issue with RowMajor vs ColMajor storage order!" << endl << endl;
                return false;
            }

            // Crout Decomposition
            Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> LU(mat_size, mat_size);
            Matrix<T, Dynamic, 1> crout_solution(mat_size);
            
            t1 = std::chrono::high_resolution_clock::now();
            Crout<T>(mat_size, A_RowMajor.data(), LU.data());
            solveCrout<T>(mat_size, LU.data(), b.data(), crout_solution.data());
            t2 = std::chrono::high_resolution_clock::now();
            crout_solve += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, crout_solution, rtol, atol)) {
                //cout << eigen_solution.transpose() << endl;
                //cout << crout_solution.transpose() << endl << endl;
                return false;
            }
        }
    }

    cout << "Eigen RowMajor : " << eigen_solve_RowMajor.count() << " s" << endl;
    cout << "Eigen ColMajor : " << eigen_solve_ColMajor.count() << " s" << endl;
    cout << "Crout          : " << crout_solve.count() << " s" << endl;

    return true;
}


int main(int argc, char** argv)
{

    cout << test_Crout<double>(1e-6, 1e-6) << endl;

    return 0;
}