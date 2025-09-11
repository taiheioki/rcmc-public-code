#ifndef RCMC_CHEMISTRY_HPP
#define RCMC_CHEMISTRY_HPP

#include <type_traits>

#include <boost/lexical_cast.hpp>

#include "network.hpp"
#include "numdef.hpp"
#include "rate_constant_matrix.hpp"

namespace rcmc
{

// Macro for defining a constant of value with type R.
#define CONSTANT(R, value) \
    ((std::is_floating_point_v<R>) ? R(value##l) : boost::lexical_cast<R>(#value))

// Returns the Boltzmann constant.
template<class R = double>
auto boltzmann_constant()
{
    return CONSTANT(R, 1.380649e-26);
}

// Returns the Planck constant.
template<class R = double>
auto planck_constant()
{
    return CONSTANT(R, 6.62607015e-37);
}

// Returns the gas constant.
template<class R = double>
auto gas_constant()
{
    return CONSTANT(R, 8.31446261815324e-3);
}

template<class R>
RateConstantMatrix<R> energy_to_rcm(const UndirectedNetwork<R>& energy_network,
                                    const R temperature,
                                    const R min_nonzero = 0.0)
{
    using boost::multiprecision::exp;
    using std::exp;

    const R gamma(1.0);

    const int n = energy_network.num_vertices();
    RateConstantMatrix<R> K(n);

    for(int i = 0; i < n; ++i) {
        K.pi(i) = exp(-energy_network.vertices[i] / (gas_constant<R>() * temperature));
    }
    const R Z = K.pi().sum();
    K.pi() /= Z;

    for(const auto& edge : energy_network.edges) {
        const int i = edge.i;
        const int j = edge.j;
        const R v   = edge.value;

        const R l_ij = -gamma * boltzmann_constant<R>() * temperature / (Z * planck_constant<R>())
                       * exp(-v / (gas_constant<R>() * temperature));

        if(-l_ij >= min_nonzero) {
            K.L(i, j) += l_ij;
            K.L(j, i) += l_ij;
        }
        else {
            std::cout << "Removed rate constant: i = " << i << ", j = " << j
                      << ", L_{ij} = " << l_ij << std::endl;
        }
    }

    for(int i = 0; i < n; ++i) {
        K.L(i, i) = -K.L().col(i).sum();
    }

    return K;
}

}  // namespace rcmc

#endif
