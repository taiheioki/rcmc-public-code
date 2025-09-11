#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "chemistry.hpp"
#include "io.hpp"
#include "linalg.hpp"
#include "rcmc.hpp"

template<class R>
void run(const std::string& data)
{
    // Load data
    const auto rcm_path    = rcmc::DATA_DIR / (data + ".txt");
    const auto rcm_network = rcmc::load_network<R>(rcm_path);
    const auto rcm = rcmc::RateConstantMatrix<R>::from_network(rcm_network);
    std::cout << "Loaded " << rcm_path << "." << std::endl;

    // Print data information
    const rcmc::SparseMatrix<R> K = rcm.K();
    const rcmc::Vector<R> pi      = rcm.pi();
    const int n                   = K.rows();
    std::cout << "#EQ = " << n << std::endl;
    std::cout << "#TS = " << rcm.num_ts() << std::endl;
    std::cout << std::endl;

    // Compute steady variables
    std::cout << "Computing epochs..." << std::endl;
    rcmc::Rcmc<R> rcmc(K, pi);
    const auto epochs = rcmc.compute_epochs();

    // Save the population history
    const auto precision = std::to_string(std::numeric_limits<R>::digits10);
    const auto output_path =
        rcmc::RESULT_DIR / (data + "-epoch-" + precision + ".csv");
    rcmc::save_epochs<R>(epochs, output_path);
    std::cout << "Saved epochs to " << output_path << "." << std::endl;
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
        ("precision,p", po::value(&precision)->default_value(15), "Precision")
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
