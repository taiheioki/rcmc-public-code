#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

#include "numdef.hpp"

namespace rcmc
{
// Set each diagonals to the minus of the sum of off-diagonals in each column.
// Original diagonals are ignored.
template<class R>
inline void set_diagonals(Matrix<R>& A)
{
    assert(A.rows() == A.cols());
    const int n = A.rows();

    for(int i = 0; i < n; ++i) {
        A(i, i) = 0.0;
        A(i, i) = -A.col(i).sum();
    }
}

// Count the number of nonzero entries
template<class Derived>
inline int nnz(const Eigen::MatrixBase<Derived>& A)
{
    return (A.array() != 0.0).count();
}

// Count the number of nonzero entries
template<class R>
inline int nnz(const SparseMatrix<R>& A)
{
    return A.nonZeros();
}

// e_i \in R^n
template<class R>
inline Vector<R> standard_vector(const int n, const int i)
{
    assert(0 <= i && i < n);
    Vector<R> e_i = Vector<R>::Zero(n);
    e_i(i)        = 1.0;
    return e_i;
}

// Returns the l-infinity norm (the maximum absolute value of the entries) of a matrix
template<class Derived>
inline typename Eigen::MatrixBase<Derived>::Scalar inf_norm(const Eigen::MatrixBase<Derived>& A)
{
    if(A.size() == 0) {
        return 0.0;
    }
    else {
        return A.cwiseAbs().maxCoeff();
    }
}

template<class Derived>
inline typename Eigen::MatrixBase<Derived>::Scalar min_nonzero_abs(
    const Eigen::MatrixBase<Derived>& A)
{
    using boost::multiprecision::fabs;
    using std::fabs;

    auto result = std::numeric_limits<typename Eigen::MatrixBase<Derived>::Scalar>::infinity();

    for(Eigen::Index j = 0; j < A.cols(); ++j) {
        for(Eigen::Index i = 0; i < A.rows(); ++i) {
            if(A(i, j) != 0.0 && result > fabs(A(i, j))) {
                result = fabs(A(i, j));
            }
        }
    }

    return result;
}

}

#endif
