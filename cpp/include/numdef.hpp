#ifndef RCMC_NUMDEF_HPP
#define RCMC_NUMDEF_HPP

// https://github.com/microsoft/vscode-cpptools/issues/7413#issuecomment-827172897
#if __INTELLISENSE__
    #undef __ARM_NEON
    #undef __ARM_NEON__
#endif

#include <sstream>
#include <string>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/multiprecision/gmp.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace rcmc
{
template<class R>
using Matrix = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>;

template<class R>
using Vector = Eigen::Matrix<R, Eigen::Dynamic, 1>;

template<class R>
using RowVector = Eigen::Matrix<R, 1, Eigen::Dynamic>;

template<class R>
using SparseMatrix = Eigen::SparseMatrix<R>;

template<class R>
using SparseVector = Eigen::SparseVector<R>;

template<unsigned int N>
using GmpFloat =
    boost::multiprecision::number<boost::multiprecision::gmp_float<N>>;


template<class R>
inline R div(const R& a, const R& b)
{
    return b == 0.0 ? std::numeric_limits<R>::infinity() : a / b;
}

template<unsigned int N>
inline GmpFloat<N> div(const GmpFloat<N>& a, const GmpFloat<N>& b)
{
    return b == 0.0 ? std::numeric_limits<GmpFloat<N>>::max() : a / b;
}

template<class R>
inline R inv(const R& a)
{
    return div(R(1.0), a);
}

template<class R>
inline void set_inf(R& val)
{
    val = std::numeric_limits<R>::infinity();
}

template<unsigned int N>
inline void set_inf(GmpFloat<N>& val)
{
    val = std::numeric_limits<GmpFloat<N>>::max();
}

template<class R>
inline bool isinf(const R& val) {
    return std::isinf(val);
}

template<unsigned int N>
inline bool isinf(const GmpFloat<N>& val) {
    return val == std::numeric_limits<GmpFloat<N>>::max();
}

}  // namespace rcmc

#endif
