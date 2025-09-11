#ifndef RCMC_RATE_CONSTANT_MATRIX_HPP
#define RCMC_RATE_CONSTANT_MATRIX_HPP

#include <vector>

#include "network.hpp"
#include "numdef.hpp"
#include "utility.hpp"

namespace rcmc
{

template<class R>
class RateConstantMatrix
{
private:
    SparseMatrix<R> m_L;
    Vector<R> m_pi;

public:
    explicit RateConstantMatrix(const int n) : m_L(n, n), m_pi(Vector<R>::Zero(n)) {}

    RateConstantMatrix(const SparseMatrix<R>& L, const Vector<R>& pi) : m_L(L), m_pi(pi) {}

    SparseMatrix<R>& L()
    {
        return m_L;
    }

    const SparseMatrix<R>& L() const
    {
        return m_L;
    }

    R& L(const int i, const int j)
    {
        return L().coeffRef(i, j);
    }

    R L(const int i, const int j) const
    {
        return L().coeff(i, j);
    }

    Vector<R>& pi()
    {
        return m_pi;
    }

    const Vector<R>& pi() const
    {
        return m_pi;
    }

    R& pi(const int i)
    {
        return pi()(i);
    }

    R pi(const int i) const
    {
        return pi()(i);
    }

    auto Pi() const
    {
        return pi().asDiagonal();
    }

    auto K() const
    {
        return -L() * Pi().inverse();
    }

    R K(const int i, const int j) const
    {
        return -L(i, j) / pi(j);
    }

    int num_eq() const
    {
        return pi().size();
    }

    int num_ts() const
    {
        return (nnz(L()) - num_eq()) / 2;
    }

    RateConstantMatrix<R> permuted(const Eigen::PermutationMatrix<Eigen::Dynamic>& P) const
    {
        return RateConstantMatrix(P * L() * P.inverse(), P * pi());
    }

    static RateConstantMatrix<R> from_network(const UndirectedNetwork<R>& N)
    {
        const int n = N.num_vertices();
        RateConstantMatrix<R> K(n);

        for(int i = 0; i < n; ++i) {
            K.pi(i) = N.vertices[i];
        }

        std::vector<Eigen::Triplet<R>> triplets;
        triplets.reserve(N.num_edges() * 2);

        for(const auto& edge : N.edges) {
            const int i = edge.i;
            const int j = edge.j;
            const R v   = edge.value;

            triplets.emplace_back(i, j, v);
            if(i != j) {
                triplets.emplace_back(j, i, v);
            }
        }

        K.L().setFromTriplets(triplets.begin(), triplets.end());

        return K;
    }

    UndirectedNetwork<R> to_network() const
    {
        const int n = num_eq();
        UndirectedNetwork<R> N(n);

        for(int i = 0; i < n; ++i) {
            N.vertices[i] = pi(i);
        }

        for(int k = 0; k < n; ++k) {
            for(typename SparseMatrix<R>::InnerIterator it(L(), k); it; ++it) {
                const int i = it.row();
                const int j = it.col();
                if(i <= j) {
                    N.add_edge(i, j, it.value());
                }
            }
        }

        return N;
    }
};
}  // namespace rcmc

#endif
