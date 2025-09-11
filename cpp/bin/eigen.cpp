#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "numdef.hpp"

#include <boost/program_options.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include "chemistry.hpp"
#include "io.hpp"
#include "rcmc.hpp"

template<class R>
void run(const std::string& data)
{
    // Load data
    const auto input_path = rcmc::DATA_DIR / (data + ".txt");
    const auto N          = rcmc::load_network<R>(input_path);
    const auto K          = rcmc::RateConstantMatrix<R>::from_network(N);
    std::cout << "Loaded " << input_path << "." << std::endl;

    const rcmc::Vector<R> sqrt_pi     = K.pi().array().sqrt();
    const rcmc::Vector<R> inv_sqrt_pi = R(1.0) / K.pi().array().sqrt();
    const rcmc::SparseMatrix<R> S =
        inv_sqrt_pi.asDiagonal() * K.L() * inv_sqrt_pi.asDiagonal();

    // Compute eigenvalues and eigenvectors
    const int n = K.num_eq();
    Spectra::SparseSymMatProd<R> op(S);
    Spectra::SymEigsSolver solver(op, n - 1, n, true);

    solver.init();
    solver.compute(Spectra::SortRule::LargestAlge);

    // Retrieve results
    std::cout << "Retriving the eigenvalues" << std::endl;
    rcmc::Vector<R> lambda(n);
    lambda << -solver.eigenvalues(), R(0.0);

    std::cout << "Retriving the eigenvectors" << std::endl;
    rcmc::Matrix<R> U = solver.eigenvectors();

    std::cout << "Converting the eigenvectors from `S` to `K`" << std::endl;
    U = sqrt_pi.asDiagonal() * U;
    U.conservativeResize(n, n);
    U.col(n - 1) = K.pi();

    // Save results
    std::cout << "Saving the result" << std::endl;
    const auto path =
        rcmc::RESULT_DIR
        / (data + "-eig-" + std::to_string(std::numeric_limits<R>::digits10)
           + ".txt");
    rcmc::save_eigen(lambda, U, path);
}

int main(const int argc, const char* const* const argv)
{
    namespace po = boost::program_options;

    std::string data;
    int precision;
    po::options_description description("Options");

    // clang-format off
    description.add_options()
        ("data", po::value(&data), "Input data name")
        ("precision,p", po::value(&precision)->default_value(200), "Precision")
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

    if(precision == 15) {
        run<double>(data);
    }
    else if(precision == 50) {
        run<rcmc::GmpFloat<50>>(data);
    }
    else if(precision == 100) {
        run<rcmc::GmpFloat<100>>(data);
    }
    else if(precision == 200) {
        run<rcmc::GmpFloat<200>>(data);
    }
    else if(precision == 400) {
        run<rcmc::GmpFloat<400>>(data);
    }
    else {
        std::cerr << "Unsupported precision: " << precision << std::endl;
        return EXIT_FAILURE;
    }
}
