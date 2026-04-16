cmake_minimum_required(VERSION 3.15)
project(GBM_Path_Simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library (shared across simulator and tests)
add_library(gbm_core STATIC
    src/cholesky.cpp
    src/statistics.cpp
    src/convergence_runner.cpp
)

# Main executable
add_executable(gbm_simulator main.cpp)
target_link_libraries(gbm_simulator PRIVATE gbm_core)

# Tests
enable_testing()

add_executable(test_integrator tests/test_integrator.cpp)

add_executable(test_cholesky tests/test_cholesky.cpp)
target_link_libraries(test_cholesky PRIVATE gbm_core)

add_executable(test_statistics tests/test_statistics.cpp)
target_link_libraries(test_statistics PRIVATE gbm_core)

add_test(NAME TestIntegrator COMMAND test_integrator)
add_test(NAME TestCholesky COMMAND test_cholesky)
add_test(NAME TestStatistics COMMAND test_statistics)
