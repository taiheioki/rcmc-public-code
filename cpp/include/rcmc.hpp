#ifndef RCMC_RCMC_HPP
#define RCMC_RCMC_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/timer/progress_display.hpp>
#include <boost/timer/timer.hpp>

#include <Eigen/LU>

#include "linalg.hpp"
#include "numdef.hpp"
#include "rate_constant_matrix.hpp"

namespace rcmc
{

template<class R>
class Rcmc
{
public:
    struct Epoch
    {
        int eq;
        R time_diag;
        R time_eig;
        R time_gershgorin;
        R rho_D;
        R rho_D_gershgorion_row;
        R rho_D_gershgorion_col;
        R sigma_KSS;
        R sigma_KSS_gershgorin_row;
        R sigma_KSS_gershgorin_col;
    };

private:
    int n;  // Number of EQs
    int num_S;
    int num_T;

    SparseMatrix<R> K;  // Rate constant matrix
    Vector<R> pi;       // Boltzmann distribution
    Eigen::PermutationMatrix<Eigen::Dynamic>
        P;  // Permutation to reorder EQs to be selected
    Eigen::PermutationMatrix<Eigen::Dynamic> P_inv;  // Inverse of `P`
    Matrix<R> C;                                     // Cholesky factor of `K`
    Matrix<R> D;  // The Schur complement of `S` in `K`

    std::string type;

    std::vector<Epoch> epochs;  // Epochs
    std::vector<Vector<R>> qs;  // Population at each epoch

public:
    explicit Rcmc(const SparseMatrix<R>& K, const Vector<R>& pi)
      : n(K.rows()), num_S(0), num_T(K.rows()), K(K), pi(pi)
    {}

private:
    // Computes K_SS.inverse() * b
    template<class Derived>
    Vector<R> K_SS_inv(const Eigen::MatrixBase<Derived>& b) const
    {
        assert(b.rows() == num_S);
        const auto L = C_SS().template triangularView<Eigen::Lower>();
        const auto U =
            C_SS().transpose().template triangularView<Eigen::Upper>();
        return -1.0 * Pi_S() * U.solve(L.solve(b));
    }

    auto K_ST() const
    {
        return K.topRightCorner(num_S, num_T);
    }

    auto K_TS() const
    {
        return K.bottomLeftCorner(num_T, num_S);
    }

    auto pi_S() const
    {
        return pi.head(num_S);
    }

    auto Pi_S() const
    {
        return pi_S().asDiagonal();
    }

    auto pi_T() const
    {
        return pi.tail(num_T);
    }

    auto Pi_T() const
    {
        return pi_T().asDiagonal();
    }

    auto C_SS() const
    {
        return C.topLeftCorner(num_S, num_S);
    }

    auto D_TT() const
    {
        return D.bottomRightCorner(num_T, num_T);
    }

    R compute_sigma_KSS() const
    {
        return inv(spectral_radius(SelfAdjointMatInv(C_SS(), pi_S())));
    }

    R compute_rho_D() const
    {
        return spectral_radius(SelfAdjointMatProd(D_TT(), pi_T()));
    }

    void greedy(bool compute_eigen, bool compute_gershgorin)
    {
        using boost::multiprecision::log;
        using boost::multiprecision::sqrt;
        using std::log;
        using std::sqrt;

        num_S = 0;
        num_T = n;

        P = Eigen::PermutationMatrix<Eigen::Dynamic>(n);
        P.setIdentity();
        P_inv = P;

        C = Matrix<R>::Zero(n, n);
        D = K;

        // `row_abs_sum(i)` is the sum of the absolute values of the entries in
        // the `i`th row of `A`
        Vector<R> row_abs_sum = D.cwiseAbs().rowwise().sum();

        epochs.resize(n);
        epochs[0].eq        = -1;
        epochs[0].time_diag = 0.0;
        qs.resize(n);

        const R log_2 = log(R(2.0));

        for(int k = 0; k < n; ++k) {
            // Compute the spectral radius
            if(compute_eigen) {
                epochs[k].rho_D     = compute_rho_D();
                epochs[k].sigma_KSS = compute_sigma_KSS();
                if(isinf(epochs[k].sigma_KSS)) {
                    set_inf(epochs[k].time_eig);
                }
                else {
                    epochs[k].time_eig = div(
                        log_2, R(sqrt(epochs[k].rho_D * epochs[k].sigma_KSS)));
                }
            }

            if(compute_gershgorin) {
                epochs[k].rho_D_gershgorion_row =
                    inf_norm(row_abs_sum.tail(n - k));
                epochs[k].rho_D_gershgorion_col =
                    2.0 * inf_norm(D_TT().diagonal());
                const R rho_D_gershgorin =
                    std::min(epochs[k].rho_D_gershgorion_row,
                             epochs[k].rho_D_gershgorion_col);

                epochs[k].sigma_KSS_gershgorin_row =
                    inv(inf_norm(K_SS_inv(Vector<R>::Ones(k, 1))));
                epochs[k].sigma_KSS_gershgorin_col =
                    inv(inf_norm(K_SS_inv(pi_S()).cwiseQuotient(pi_S())));
                const R sigma_KSS_gershgorin =
                    std::max(epochs[k].sigma_KSS_gershgorin_row,
                             epochs[k].sigma_KSS_gershgorin_col);

                if(isinf(sigma_KSS_gershgorin)) {
                    set_inf(epochs[k].time_gershgorin);
                }
                else {
                    epochs[k].time_gershgorin =
                        div(log_2,
                            R(sqrt(rho_D_gershgorin * sigma_KSS_gershgorin)));
                }
            }

            // Find the largest entry
            R largest = 0.0;
            int pivot = -1;
            for(int j = k; j < n; ++j) {
                const R value = -D(j, j);
                if(largest < value) {
                    largest = value;
                    pivot   = j;
                }
            }
            if(pivot == -1 || k == n - 1) {
                assert(pivot == -1 && k == n - 1);
                break;
            }

            // Swap the k-th and pivot-th rows and columns
            std::swap(pi(k), pi(pivot));
            C.row(k).head(k).swap(C.row(pivot).head(k));

            // Swap entire rows for later use in Type B
            // In Type A, `D.row(k).tail(n - k).swap(D.row(pivot).tail(n - k));`
            // is enough
            D.row(k).swap(D.row(pivot));
            D.col(k).tail(n - k).swap(D.col(pivot).tail(n - k));

            std::swap(row_abs_sum(k), row_abs_sum(pivot));
            P.applyTranspositionOnTheLeft(k, pivot);
            P_inv.applyTranspositionOnTheRight(k, pivot);

            // Eliminate each column
            for(int j = k + 1; j < n; ++j) {
                if(D(k, j) != 0.0) {
                    D.col(j).tail(n - k - 1) -=
                        D(k, j) / D(k, k) * D.col(k).tail(n - k - 1);
                    D(j, j) = 0.0;

                    if(compute_gershgorin) {
                        const auto pi_T = pi.tail(n - k - 1);
                        row_abs_sum(j)  = pi(j)
                                         * D.col(j)
                                               .tail(n - k - 1)
                                               .cwiseQuotient(pi_T)
                                               .sum();
                    }

                    D(j, j) = -D.col(j).tail(n - k - 1).sum();

                    if(compute_gershgorin) {
                        row_abs_sum(j) -= D(j, j);
                    }
                }
            }

            // Change the `k`th column of `C` to the Cholesky factor
            C.col(k).tail(n - k) =
                -sqrt(-pi(k) / D(k, k)) * D.col(k).tail(n - k);

            epochs[k + 1].eq        = P_inv.indices()[k];
            epochs[k + 1].time_diag = 1.0 / largest;

            ++num_S;
            --num_T;
        }

        K = K.twistedBy(P);
    }

    void compute_rcmc(Vector<R> p, const std::string& type = "A")
    {
        num_S = 0;
        num_T = n;
        p     = P * p;

        // Cholesky factor of `M = I_T + K_ST * K_SS^{-2} * K_TS`
        Matrix<R> G;
        if(type == "B") {
            G = Vector<R>(pi.array().sqrt()).asDiagonal();
        }

        for(int k = 0; k < n; ++k) {
            const auto p_S = p.head(num_S);
            const auto p_T = p.tail(num_T);

            Vector<R> q(n);
            auto q_S = q.head(num_S);
            auto q_T = q.tail(num_T);

            if(type == "A") {
                q_T = (Vector<R>::Ones(num_T, 1)
                       - Pi_T().inverse() * K_TS() * K_SS_inv(pi_S()))
                          .asDiagonal()
                          .inverse()
                      * (p_T - K_TS() * K_SS_inv(p_S));
            }
            else if(type == "B") {
                const auto G_TT = G.bottomRightCorner(num_T, num_T);
                const auto L    = G_TT.template triangularView<Eigen::Lower>();
                const auto U =
                    G_TT.transpose().template triangularView<Eigen::Upper>();
                q_T = Pi_T() * U.solve(L.solve(p_T - K_TS() * K_SS_inv(p_S)));
            }
            else {
                std::cerr << "Invalid type: " << type << std::endl;
                std::exit(EXIT_FAILURE);
            }

            q_S = -1.0 * K_SS_inv(K_ST() * q_T);

            if(type == "B") {
                project(q);
            }

            qs[k] = P_inv * q;

            ++num_S;
            --num_T;

            if(type == "B") {
                cholupdate(G.bottomRightCorner(num_T, num_T),
                           G.col(k).tail(num_T)
                               - G(k, k) / D(k, k) * D.col(k).tail(num_T));
            }
        }
    }

    void project(Vector<R>& w)
    {
        assert(w.size() == n);

        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i1, int i2) {
            return w(i1) / pi(i1) > w(i2) / pi(i2);
        });

        R cum_pi = 0.0;

        // For numerical stability, we calculate
        // `cum_w[j] = w[idx[j + 1]] + ... + w[idx[n - 1]] = 1 - (w[idx[0]] +
        // ... + w[idx[j]])`
        std::vector<R> cum_w(n);
        cum_w[n - 1] = 0.0;
        for(int j = n - 1; j > 0; --j) {
            cum_w[j - 1] = cum_w[j] + w(idx[j]);
        }

        int l = n - 1;
        for(int j = 0; j < n; ++j) {
            const R w_j  = w(idx[j]);
            const R pi_j = pi(idx[j]);
            if(w_j + pi_j * cum_w[j] / (cum_pi + pi_j) <= 0.0) {
                l = j - 1;
                break;
            }
            cum_pi += pi_j;
        }

        assert(l != -1);
        const R mu = cum_w[l] / cum_pi;

        for(int i = 0; i < n; ++i) {
            w(i) = std::max(R(w(i) + pi(i) * mu), R(0.0));
        }
    }

public:
    const std::vector<Epoch>& compute_epochs()
    {
        boost::timer::auto_cpu_timer timer;
        greedy(true, true);
        return epochs;
    }

    const std::vector<Vector<R>>& compute_populations(
        Vector<R> p,
        const std::string& type = "A",
        const std::string& time = "diag")
    {
        boost::timer::cpu_timer timer;
        greedy(time == "eigen", time == "gershgorin");
        compute_rcmc(p, type);

        const auto elapsed = timer.elapsed();
        std::cout << "Running time: "
                  << static_cast<double>(elapsed.wall) * 1e-9 << " sec"
                  << std::endl;

        return qs;
    }
};

}  // namespace rcmc

#endif
