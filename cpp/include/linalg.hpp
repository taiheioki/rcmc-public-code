#ifndef RCMC_LINALG_HPP
#define RCMC_LINALG_HPP

#include <algorithm>
#include <cstdlib>
#include <exception>

#include <Spectra/SymEigsSolver.h>

#include "utility.hpp"

namespace rcmc
{

// Compute the largest eigenvalue in the absolute value of a self-adjoint matrix.
template<class Op>
typename Op::Scalar spectral_radius(Op op)
{
    using boost::multiprecision::fabs;
    using std::fabs;

    const int n = op.rows();
    if(n == 0) {
        return 0.0;
    }
    else if(n == 1) {
        return fabs(op.scalar());
    }

    Spectra::SymEigsSolver solver(op, 1, std::min(n, 32), false);
    solver.init();

    try {
        if(solver.compute() == 0) {
            std::cerr << "Spectra::SymEigsSolver failed to converge." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Spectra::SymEigsSolver failed: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return fabs(solver.eigenvalues()(0));
}

// Computes `symmetrize(A) * x` for a self-adjoint matrix `A`.
template<class DerivedA, class DerivedPi>
class SelfAdjointMatProd
{
public:
    using Scalar = typename Eigen::MatrixBase<DerivedA>::Scalar;

private:
    const Eigen::MatrixBase<DerivedA>& A;
    const Eigen::MatrixBase<DerivedPi>& pi;
    SparseMatrix<Scalar> A_sparse;
    bool is_sparse;

public:
    explicit SelfAdjointMatProd(const Eigen::MatrixBase<DerivedA>& A,
                                const Eigen::MatrixBase<DerivedPi>& pi)
      : A(A), pi(pi)
    {
        if(nnz(A) < A.size() / 3) {
            A_sparse  = A.sparseView();
            is_sparse = true;
        }
        else {
            is_sparse = false;
        }
    }

    Eigen::Index rows() const
    {
        return pi.size();
    }

    Eigen::Index cols() const
    {
        return pi.size();
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        const auto Pi_sqrt     = pi.cwiseSqrt().asDiagonal();
        const auto Pi_sqrt_inv = pi.cwiseSqrt().cwiseInverse().asDiagonal();

        Eigen::Map<const Vector<Scalar>> x(x_in, cols());
        Eigen::Map<Vector<Scalar>> y(y_out, rows());

        if(is_sparse) {
            const auto S = Pi_sqrt_inv * A_sparse * Pi_sqrt;
            y.noalias()  = S.template selfadjointView<Eigen::Lower>() * x;
        }
        else {
            const auto S = Pi_sqrt_inv * A * Pi_sqrt;
            y.noalias()  = S.template selfadjointView<Eigen::Lower>() * x;
        }
    }

    Scalar scalar() const
    {
        assert(rows() == 1);
        if(is_sparse) {
            return A_sparse.coeff(0, 0);
        }
        else {
            return A(0, 0);
        }
    }
};

// Computes `symmetrize(A).inverse() * x`,
// where `A = -C * C.transpose() * pi.asDiagonal().inverse()` is self-adjoint
template<class DerivedC, class DerivedPi>
class SelfAdjointMatInv
{
private:
    const Eigen::MatrixBase<DerivedC>& C;
    const Eigen::MatrixBase<DerivedPi>& pi;

public:
    using Scalar = typename Eigen::MatrixBase<DerivedC>::Scalar;

    explicit SelfAdjointMatInv(const Eigen::MatrixBase<DerivedC>& C,
                               const Eigen::MatrixBase<DerivedPi>& pi)
      : C(C), pi(pi)
    {}

    Eigen::Index rows() const
    {
        return pi.size();
    }

    Eigen::Index cols() const
    {
        return pi.size();
    }

    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        const auto L       = C.template triangularView<Eigen::Lower>();
        const auto U       = C.transpose().template triangularView<Eigen::Upper>();
        const auto Pi_sqrt = pi.cwiseSqrt().asDiagonal();

        Eigen::Map<const Vector<Scalar>> x(x_in, cols());
        Eigen::Map<Vector<Scalar>> y(y_out, rows());
        y.noalias() = -1.0 * Pi_sqrt * U.solve(L.solve(Pi_sqrt * x));
    }

    Scalar scalar() const
    {
        assert(rows() == 1);
        return -pi(0) / (C(0, 0) * C(0, 0));
    }
};

template<class Derived>
void cholupdate(
    typename Eigen::Block<Derived> L,
    Vector<typename Eigen::Block<Derived>::Scalar> x,
    const bool plus = true
){
    using std::sqrt;
    using boost::multiprecision::sqrt;
    using R = typename Eigen::Block<Derived>::Scalar;

    const int n = x.size();
    assert(L.rows() == n && L.cols() == n);

    for (int i = 0; i < n; i++) {
        const R r = sqrt(L(i, i) * L(i, i) + (plus ? 1.0 : -1.0) * x(i) * x(i));
        const R c = r / L(i, i);
        const R s = x(i) / L(i, i);
        L(i, i) = r;

        if(i < n - 1) {
            auto L_ip = L.block(i + 1, i, n - i - 1, 1);
            auto x_ip = x.segment(i + 1, n - i - 1);
            L_ip = (L_ip + (plus ? 1.0 : -1.0) * s * x_ip) / c;
            x_ip = c * x_ip - s * L_ip;
        }
    }
}

}  // namespace rcmc

#endif
