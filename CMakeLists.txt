cmake_minimum_required(VERSION 3.15)
project(GBM_Path_Simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main executable
add_executable(gbm_simulator
    main.cpp
    src/cholesky.cpp
    src/statistics.cpp
    src/convergence_runner.cpp
)

# Test executables
add_executable(test_integrator tests/test_integrator.cpp)
add_executable(test_cholesky tests/test_cholesky.cpp src/cholesky.cpp)
add_executable(test_statistics tests/test_statistics.cpp src/statistics.cpp src/cholesky.cpp)
