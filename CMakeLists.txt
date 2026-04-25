cmake_minimum_required(VERSION 3.15)
project(GBM_Path_Simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers (top-level include/ and include/bs/ are both reachable)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/bs)

# ── GBM simulation core ──────────────────────────────────────────────────────
add_library(gbm_core STATIC
    src/cholesky.cpp
    src/statistics.cpp
    src/convergence_runner.cpp
    src/time_grid.cpp
    src/rng.cpp
)

# ── Black-Scholes option pricing library ────────────────────────────────────
add_library(bs_pricing STATIC
    src/bs/normal.cpp
    src/bs/payoff.cpp
    src/bs/black_scholes.cpp
)

# ── Hedging Backtester library ──────────────────────────────────────────────
add_library(hedger_core STATIC
    src/hedging/delta_hedger.cpp
)
target_link_libraries(hedger_core PRIVATE bs_pricing)

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(gbm_simulator main.cpp)
target_link_libraries(gbm_simulator PRIVATE gbm_core bs_pricing hedger_core)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_integrator tests/test_integrator.cpp)

add_executable(test_cholesky tests/test_cholesky.cpp)
target_link_libraries(test_cholesky PRIVATE gbm_core)

add_executable(test_statistics tests/test_statistics.cpp)
target_link_libraries(test_statistics PRIVATE gbm_core)

add_executable(test_black_scholes tests/test_black_scholes.cpp)
target_link_libraries(test_black_scholes PRIVATE bs_pricing)

add_executable(test_hedging tests/test_hedging.cpp)
target_link_libraries(test_hedging PRIVATE hedger_core gbm_core bs_pricing)

add_test(NAME TestIntegrator    COMMAND test_integrator)
add_test(NAME TestCholesky      COMMAND test_cholesky)
add_test(NAME TestStatistics    COMMAND test_statistics)
add_test(NAME TestBlackScholes  COMMAND test_black_scholes)
add_test(NAME TestHedging       COMMAND test_hedging)
