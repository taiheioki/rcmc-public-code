#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/LU>
#include <boost/program_options.hpp>

#include "chemistry.hpp"
#include "io.hpp"
#include "linalg.hpp"
#include "rcmc.hpp"

template<class R>
void run(const std::string& data,
         const std::string& type,
         const std::string& time)
{
    // Load data
    const auto rcm_path    = rcmc::DATA_DIR / (data + ".txt");
    const auto rcm_network = rcmc::load_network<R>(rcm_path);
    const auto rcm = rcmc::RateConstantMatrix<R>::from_network(rcm_network);
    std::cout << "Loaded " << rcm_path << "." << std::endl;

    const rcmc::SparseMatrix<R> K = rcm.K();
    const rcmc::Vector<R> pi      = rcm.pi();
    const int n                   = K.rows();

    // Compute steady variables
    std::cout << "Running the RCMC method..." << std::endl;
    rcmc::Rcmc<R> rcmc(K, pi);
    const auto populations =
        rcmc.compute_populations(rcmc::standard_vector<R>(n, 0), type, time);

    // Save the population history
    const auto output_path =
        rcmc::RESULT_DIR
        / (data + "-pop-" + type + "-"
           + std::to_string(std::numeric_limits<R>::digits10) + ".txt");
    rcmc::save_populations(populations, output_path);
    std::cout << "Saved results to " << output_path << "." << std::endl;
}

int main(const int argc, const char* const* const argv)
{
    namespace po = boost::program_options;

    std::string data;
    std::string type;
    std::string time;
    int precision;
    po::options_description description("Options");

    // clang-format off
    description.add_options()
        ("data", po::value(&data), "Input data name")
        ("type,t", po::value(&type)->default_value("A"), "RCMC Type (A or B)")
        ("reference,r", po::value(&time)->default_value("diag"), "Method of computing reference time (diag, eigen, or gershgorin)")
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
        run<double>(data, type, time);
    }
    else if(precision == 50) {
        run<rcmc::GmpFloat<50>>(data, type, time);
    }
    else if(precision == 100) {
        run<rcmc::GmpFloat<100>>(data, type, time);
    }
    else if(precision == 200) {
        run<rcmc::GmpFloat<200>>(data, type, time);
    }
    else if(precision == 400) {
        run<rcmc::GmpFloat<400>>(data, type, time);
    }
    else {
        std::cerr << "Unsupported precision: " << precision << std::endl;
        return EXIT_FAILURE;
    }
}
