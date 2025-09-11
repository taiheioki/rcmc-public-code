cmake_minimum_required(VERSION 3.20)
project(rcmc
    VERSION 1.0
    DESCRIPTION "Code for numerical experiments in the paper \"Rate Constant Matrix Contraction Method for Stiff Master Equations with Detailed Balance\""
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

find_package(Boost 1.7 REQUIRED COMPONENTS program_options timer)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED gmp gmpxx IMPORTED_TARGET)

include(FetchContent)
FetchContent_Declare(
    Spectra
    GIT_REPOSITORY git@github.com:taiheioki/spectra.git
    GIT_TAG master
)
FetchContent_MakeAvailable(Spectra)

add_library(rcmc INTERFACE)

target_include_directories(rcmc INTERFACE include)
target_compile_features(rcmc INTERFACE cxx_std_17)
target_compile_options(rcmc INTERFACE -Wall -Wextra)
target_link_libraries(rcmc INTERFACE Boost::headers Boost::program_options Boost::timer Eigen3::Eigen PkgConfig::GMP Spectra)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(pop bin/pop.cpp)
target_link_libraries(pop rcmc)

add_executable(eigen bin/eigen.cpp)
target_link_libraries(eigen rcmc)

add_executable(ode bin/ode.cpp)
target_link_libraries(ode rcmc)

add_executable(epoch bin/epoch.cpp)
target_link_libraries(epoch rcmc)
