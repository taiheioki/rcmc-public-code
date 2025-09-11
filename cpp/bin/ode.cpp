#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "chemistry.hpp"
#include "io.hpp"
#include "rcmc.hpp"

// Generate a vector of `N` logarithmically spaced numbers from `t_min` to
// `t_max`.
template<class R>
std::vector<R> logspace(const R t_min, const R t_max, const int N)
{
    using boost::multiprecision::log10;
    using boost::multiprecision::pow;
    using std::log10;
    using std::pow;

    std::vector<R> result(N);
    const R base = pow(R(10.0), log10(t_max / t_min) / (N - 1));

    for(int k = 0; k < N; k++) {
        result[k] = t_min * pow(base, k);
    }

    return result;
}

template<class R, class ER>
void run(const std::string& data, bool load_epoch)
{
    using boost::multiprecision::fabs;
    using boost::multiprecision::max;
    using std::fabs;
    using std::max;

    // Load RCM
    const auto rcm_path = rcmc::DATA_DIR / (data + ".txt");
    const auto K        = rcmc::RateConstantMatrix<R>::from_network(
        rcmc::load_network<R>(rcm_path));
    std::cout << "Loaded " << rcm_path << "." << std::endl;

    // Load eigenvalues and eigenvectors
    const auto precision = std::to_string(std::numeric_limits<R>::digits10);
    const auto eigen_path =
        rcmc::RESULT_DIR / (data + "-eig-" + precision + ".txt");
    const auto [lambda, U] = rcmc::load_eigen<R>(eigen_path);
    std::cout << "Loaded " << eigen_path << "." << std::endl;

    // Compute the coefficients
    const int n                = K.num_eq();
    const auto x_0             = rcmc::standard_vector<R>(n, 0);
    const rcmc::RowVector<R> c = U.row(0) / K.pi(0);

    // Check the validity of the coefficients
    rcmc::Vector<R> x = rcmc::Vector<R>::Zero(n);
    for(int k = 0; k < n; ++k) {
        x += c(k) * U.col(k);
    }

    R max_err = fabs(R(1.0) - x(0));
    for(int i = 1; i < n; ++i) {
        max_err = max(max_err, fabs(x(i)));
    }
    std::cout << "Maximum error of the coefficients = " << max_err << std::endl;

    // Get the time points
    std::vector<R> ts;
    const auto epoch_precision =
        std::to_string(std::numeric_limits<ER>::digits10);

    if(load_epoch) {
        const auto epoch_path =
            rcmc::RESULT_DIR / (data + "-epoch-" + epoch_precision + ".csv");
        const auto epochs = rcmc::load_epochs<R>(epoch_path);

        ts.reserve(epochs.size() * 3);
        for(const auto& epoch : epochs) {
            ts.emplace_back(epoch.time_diag);
            ts.emplace_back(epoch.time_eig);
            ts.emplace_back(epoch.time_gershgorin);
        }

        std::sort(ts.begin(), ts.end());
        ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
        std::cout << "Computing " << ts.size() << " populations loaded from "
                  << epoch_path << std::endl;
    }
    else {
        const R t_min   = -1e-5 / lambda.head(n - 1).minCoeff();
        const R t_max   = -1e+5 / lambda.head(n - 1).maxCoeff();
        constexpr int N = 300;
        ts              = logspace(t_min, t_max, N);
        std::cout << "Computing " << N << " populations from t = " << t_min
                  << " to " << t_max << std::endl;
    }

    // Compute the populations
    std::vector<rcmc::Vector<R>> xs;
    xs.reserve(ts.size());
    boost::timer::progress_display progress(ts.size());

    for(const R t : ts) {
        using boost::multiprecision::exp;
        using std::exp;

        rcmc::Vector<R> x;

        if(rcmc::isinf(t)) {
            x = K.pi();
        }
        else {
            x = rcmc::Vector<R>::Zero(n);
            for(int k = 0; k < n; ++k) {
                x += c(k) * exp(lambda(k) * t) * U.col(k);
            }
        }

        xs.emplace_back(x);
        ++progress;
    }

    // Save the result
    const auto output_path =
        rcmc::RESULT_DIR
        / (data + "-ode-" + (load_epoch ? "epoch-" : "") + precision
           + (load_epoch ? "-" + epoch_precision : "") + ".csv");
    rcmc::save_population_history(ts, xs, output_path);
    std::cout << "Saved population history to " << output_path << "."
              << std::endl;
}

int main(const int argc, const char* const* const argv)
{
    namespace po = boost::program_options;

    std::string data;
    int precision;
    int epoch_precision;
    po::options_description description("Options");

    // clang-format off
    description.add_options()
        ("data", po::value(&data), "Input data name")
        ("precision,p", po::value(&precision)->default_value(200), "Precision")
        ("epoch", "Load the epoch data")
        ("epoch-precision,e", po::value(&epoch_precision)->default_value(15), "Epoch precision")
        ("help,h", "Print this help message")
    ;
    // clang-format on

    po::positional_options_description positional_description;
    positional_description.add("data", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(description)
                  .positional(positional_description)
                  .run(),
              vm);
    po::notify(vm);

    if(vm.count("help") || !vm.count("data")) {
        std::cout << description << std::endl;
        return EXIT_SUCCESS;
    }

    const bool load_epoch = vm.count("epoch");

    if(precision == 15 && epoch_precision == 15) {
        run<double, double>(data, load_epoch);
    }
    else if(precision == 50 && epoch_precision == 15) {
        run<rcmc::GmpFloat<50>, double>(data, load_epoch);
    }
    else if(precision == 100 && epoch_precision == 15) {
        run<rcmc::GmpFloat<100>, double>(data, load_epoch);
    }
    else if(precision == 200 && epoch_precision == 15) {
        run<rcmc::GmpFloat<200>, double>(data, load_epoch);
    }
    else if(precision == 400 && epoch_precision == 15) {
        run<rcmc::GmpFloat<400>, double>(data, load_epoch);
    }
    else if(precision == 800 && epoch_precision == 15) {
        run<rcmc::GmpFloat<800>, double>(data, load_epoch);
    }
    else {
        std::cerr << "Unsupported precision" << std::endl;
        return EXIT_FAILURE;
    }
}
