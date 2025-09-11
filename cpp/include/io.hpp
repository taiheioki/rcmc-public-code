#ifndef RCMC_IO_HPP
#define RCMC_IO_HPP

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <regex>
#include <utility>
#include <vector>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/timer/progress_display.hpp>

#include "network.hpp"
#include "numdef.hpp"
#include "rate_constant_matrix.hpp"
#include "rcmc.hpp"

namespace rcmc
{
const std::filesystem::path DATA_DIR   = "data";
const std::filesystem::path RESULT_DIR = "result";

// Convert std::string to T.
template<class T>
std::optional<T> from_string(const std::string& str)
{
    try {
        return boost::lexical_cast<T>(str);
    }
    catch(...) {
        return std::nullopt;
    }
}

template<>
std::optional<GmpFloat<50>> from_string<GmpFloat<50>>(const std::string& str)
{
    if(str == "inf"){
        return std::numeric_limits<GmpFloat<50>>::max();
    }
    else {
        try {
            return boost::lexical_cast<GmpFloat<50>>(str);
        }
        catch(...) {
            return std::nullopt;
        }
    }
}

template<>
std::optional<GmpFloat<100>> from_string<GmpFloat<100>>(const std::string& str)
{
    if(str == "inf"){
        return std::numeric_limits<GmpFloat<100>>::max();
    }
    else {
        try {
            return boost::lexical_cast<GmpFloat<100>>(str);
        }
        catch(...) {
            return std::nullopt;
        }
    }
}

template<>
std::optional<GmpFloat<200>> from_string<GmpFloat<200>>(const std::string& str)
{
    if(str == "inf"){
        return std::numeric_limits<GmpFloat<200>>::max();
    }
    else {
        try {
            return boost::lexical_cast<GmpFloat<200>>(str);
        }
        catch(...) {
            return std::nullopt;
        }
    }
}

template<>
std::optional<GmpFloat<400>> from_string<GmpFloat<400>>(const std::string& str)
{
    if(str == "inf"){
        return std::numeric_limits<GmpFloat<400>>::max();
    }
    else {
        try {
            return boost::lexical_cast<GmpFloat<400>>(str);
        }
        catch(...) {
            return std::nullopt;
        }
    }
}

template<class T>
T safe_from_string(const std::string& str)
{
    auto result = from_string<T>(str);
    if(!result) {
        std::cerr << "Failed to parse " << str << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return *result;
}

// Read one line from `in` and check whether the line matches to `pattern`. If
// it is, every `i`th submatch is converted to type of the `i`th template
// argument and is packed as a tuple. Returns std::nullopt when some operations
// fail.
template<class... Args>
std::optional<std::tuple<Args...>> get_line_and_parse(std::istream& in, const std::regex& pattern)
{
    std::string line;
    if(!std::getline(in, line)) {
        return std::nullopt;
    }

    std::smatch match;
    if(!std::regex_match(line, match, pattern)) {
        std::cout << "does not match" << std::endl;
        return std::nullopt;
    }

    if(match.size() < sizeof...(Args)) {
        return std::nullopt;
    }

    int i                  = 0;
    const std::tuple tuple = {[&]() { return from_string<Args>(match[++i].str()); }()...};

    if(std::apply([](auto... args) { return (args && ...); }, tuple)) {
        return std::apply([](auto... args) { return std::tuple{*args...}; }, tuple);
    }
    else {
        return std::nullopt;
    }
}

template<class R>
std::optional<UndirectedNetwork<R>> load_MinPATH(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in) {
        std::cerr << "Could not open " << path << std::endl;
        std::exit(1);
    }

    // List of EQ
    int num_eqs;
    const std::regex re_list_of_eqs(R"(List of EQs \((\d+)\):.*)");
    if(const auto tuple = get_line_and_parse<int>(in, re_list_of_eqs); tuple) {
        num_eqs = std::get<0>(*tuple);
    }
    else {
        return std::nullopt;
    }

    UndirectedNetwork<R> G(num_eqs);

    const std::regex re_eq(R"( EQ +(\d+) \( *(-?\d*\.\d*)\).*)");
    for(int k = 0; k < num_eqs; ++k) {
        const auto tuple = get_line_and_parse<int, double>(in, re_eq);
        if(!tuple) {
            return std::nullopt;
        }
        const auto [i, v] = *tuple;
        if(i < 0 || num_eqs <= i) {
            return std::nullopt;
        }
        G.vertices[i] = v;
    }

    if(!get_line_and_parse(in, std::regex(""))) {
        return std::nullopt;
    }

    // List of TSs
    int num_tss;
    const std::regex re_list_of_tss(R"(List of TSs \((\d+)\):.*)");
    if(const auto tuple = get_line_and_parse<int>(in, re_list_of_tss); tuple) {
        num_tss = std::get<0>(*tuple);
    }
    else {
        return std::nullopt;
    }

    const std::regex re_ts(R"( TS +\d+: +(-?\d+) - +(-?\d+) \( *(-?\d*\.\d*)\).*)");
    for(int k = 0; k < num_tss; ++k) {
        const auto tuple = get_line_and_parse<int, int, double>(in, re_ts);
        if(!tuple) {
            std::cerr << "Failed to parse " << path << std::endl;
            std::exit(1);
        }

        const auto [i, j, v] = *tuple;
        if(0 <= i && i < num_eqs && 0 <= j && j < num_eqs && i != j) {
            G.add_edge(i, j, v);
        }
    }

    if(!get_line_and_parse(in, std::regex(""))) {
        return std::nullopt;
    }

    // List of PTs
    int num_pts;
    const std::regex re_list_of_pts(R"(List of PTs \((\d+)\):.*)");
    if(const auto tuple = get_line_and_parse<int>(in, re_list_of_pts); tuple) {
        num_pts = std::get<0>(*tuple);
    }
    else {
        return std::nullopt;
    }

    const std::regex re_pt(R"( PT +\d+: +(-?\d+) - +(-?\d+) \( *(-?\d*\.\d*)\).*)");
    for(int k = 0; k < num_pts; ++k) {
        const auto tuple = get_line_and_parse<int, int, double>(in, re_pt);
        if(!tuple) {
            return std::nullopt;
        }

        const auto [i, j, v] = *tuple;
        if(0 <= i && i < num_eqs && 0 <= j && j < num_eqs && i != j) {
            G.add_edge(i, j, v);
        }
    }

    return G;
}

template<class R>
UndirectedNetwork<R> load_network(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;

    std::getline(in, line);
    int n = *from_string<int>(line);

    UndirectedNetwork<R> N(n);

    for(int i = 0; i < n; ++i) {
        std::getline(in, line);
        const auto v = from_string<R>(line);
        if(!v) {
            std::cerr << "Failed to parse " << path << " (i = " << i << ", line = " << line << ")" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        N.vertices[i] = *v;
    }

    std::getline(in, line);
    int m = *from_string<int>(line);

    for(int k = 0; k < m; ++k) {
        std::getline(in, line);
        std::vector<std::string> fields;
        boost::algorithm::split(fields, line, boost::is_any_of(" "));

        const auto i = from_string<int>(fields[0]);
        const auto j = from_string<int>(fields[1]);
        const auto v = from_string<R>(fields[2]);
        if(!i || !j) {
            std::cerr << "Failed to parse " << path << " (k = " << k << ", line = " << line << ")" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(v){
            N.add_edge(*i, *j, *v);
        } else {
            std::cout << "Ignored edge: i = " << *i << ", j = " << *j << ", v = " << fields[2] << std::endl;
        }
    }

    return N;
}

template<class R>
void save_network(const UndirectedNetwork<R>& N, const std::filesystem::path& path)
{
    std::filesystem::create_directory(path.parent_path());

    std::ofstream out(path);
    if(!out) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << std::scientific << std::setprecision(std::numeric_limits<R>::digits10 + 1);

    const int n = N.num_vertices();
    out << n << '\n';

    for(const R& v : N.vertices) {
        out << v << '\n';
    }

    out << N.num_edges() << '\n';

    for(const auto& edge : N.edges) {
        out << edge.i << ' ' << edge.j << ' ' << edge.value << '\n';
    }
}

template<class R>
std::pair<rcmc::Vector<R>, rcmc::Matrix<R>> load_eigen(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int n;
    in >> n;

    rcmc::Vector<R> lambda(n);
    for(int i = 0; i < n; ++i) {
        in >> lambda(i);
    }

    rcmc::Matrix<R> U(n, n);
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            in >> U(i, j);
        }
    }

    return {lambda, U};
}

template<class R>
void save_eigen(const rcmc::Vector<R>& lambda,
                const rcmc::Matrix<R>& U,
                const std::filesystem::path& path)
{
    std::filesystem::create_directory(path.parent_path());

    std::ofstream out(path);
    if(!out) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << std::scientific << std::setprecision(std::numeric_limits<R>::digits10 + 1);

    const int n = lambda.size();
    out << n << '\n';
    for(int i = 0; i < n; ++i) {
        out << lambda(i) << '\n';
    }

    boost::timer::progress_display progress(n * n);

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            out << U(i, j) << (j < n - 1 ? " " : "");
            ++progress;
        }
        out << '\n';
    }
}

template<class R>
std::vector<typename Rcmc<R>::Epoch> load_epochs(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<typename Rcmc<R>::Epoch> epochs;

    std::string line;
    std::getline(in, line);  // Skip the header line

    while(std::getline(in, line)) {
        std::vector<std::string> fields;
        boost::algorithm::split(fields, line, boost::is_any_of(","));

        typename Rcmc<R>::Epoch epoch;
        epoch.eq                       = safe_from_string<int>(fields[0]);
        epoch.time_diag                = safe_from_string<R>(fields[1]);
        epoch.time_eig                 = safe_from_string<R>(fields[2]);
        epoch.time_gershgorin          = safe_from_string<R>(fields[3]);
        epoch.rho_D                    = safe_from_string<R>(fields[4]);
        epoch.rho_D_gershgorion_row    = safe_from_string<R>(fields[5]);
        epoch.rho_D_gershgorion_col    = safe_from_string<R>(fields[6]);
        epoch.sigma_KSS                = safe_from_string<R>(fields[7]);
        epoch.sigma_KSS_gershgorin_row = safe_from_string<R>(fields[8]);
        epoch.sigma_KSS_gershgorin_col = safe_from_string<R>(fields[9]);

        epochs.push_back(epoch);
    }

    return epochs;
}

template<class R>
void save_epochs(const std::vector<typename Rcmc<R>::Epoch>& epochs,
                 const std::filesystem::path& path)
{
    std::filesystem::create_directory(path.parent_path());

    std::ofstream out(path);
    if(!out) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << std::scientific << std::setprecision(std::numeric_limits<R>::digits10 + 1);

    out << "eq,time_diag,time_eig,time_gershgorin,rho_D,rho_D_gershgorin_row,rho_D_gershgorin_col,sigma_KSS,sigma_KSS_gershgorin_row,"
           "sigma_KSS_gershgorin_col\n";

    for(const auto& epoch : epochs) {
        out << epoch.eq << ',' << epoch.time_diag << ',' << epoch.time_eig << ','
            << epoch.time_gershgorin << ',' << epoch.rho_D << ',' << epoch.rho_D_gershgorion_row
            << ',' << epoch.rho_D_gershgorion_col << ',' << epoch.sigma_KSS << ','
            << epoch.sigma_KSS_gershgorin_row << ',' << epoch.sigma_KSS_gershgorin_col << '\n';
    }
}

template<class R>
void save_populations(const std::vector<Vector<R>>& populations, const std::filesystem::path& path)
{
    std::filesystem::create_directory(path.parent_path());

    std::ofstream out(path);
    if(!out) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << std::scientific << std::setprecision(std::numeric_limits<R>::digits10 + 1);

    for(const auto& q : populations) {
        const int n = q.size();
        for(int i = 0; i < n; ++i) {
            out << q(i) << (i < n - 1 ? " " : "");
        }
        out << '\n';
    }
}

template<class R>
void save_population_history(const std::vector<R>& times,
                             const std::vector<Vector<R>>& populations,
                             const std::filesystem::path& path)
{
    std::filesystem::create_directory(path.parent_path());

    std::ofstream out(path);
    if(!out) {
        std::cerr << "File open error: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << std::scientific << std::setprecision(std::numeric_limits<R>::digits10 + 1);

    const int n = populations[0].size();

    out << "time";
    for(int i = 0; i < n; ++i) {
        out << "," << i;
    }
    out << '\n';

    for(int t = 0; t < int(times.size()); ++t) {
        out << times[t];
        for(int i = 0; i < n; ++i) {
            out << "," << populations[t](i);
        }
        out << '\n';
    }
}

}  // namespace rcmc

#endif
