cmake_minimum_required(VERSION 3.16)
project(qbam_engine LANGUAGES CXX)

# Q-BAM M1 engine (engine_contract.md). Single-threaded, C++17, hnswlib only.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Instruction-set target for -march. Default "native" preserves the frozen engine builds; override
# Override (e.g. -DQBAM_MARCH=x86-64-v3) to produce a portable binary across microarchitectures.
set(QBAM_MARCH "native" CACHE STRING "value for -march (default native)")

set(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib"
    CACHE PATH "hnswlib checkout root (must contain hnswlib/hnswlib.h)")
if(NOT EXISTS "${HNSWLIB_INCLUDE_DIR}/hnswlib/hnswlib.h")
  message(FATAL_ERROR "hnswlib not found under ${HNSWLIB_INCLUDE_DIR}/hnswlib/hnswlib.h")
endif()

set(QBAM_GIT_SHA "nogit" CACHE STRING "git commit recorded in the run manifest")
set(QBAM_BUILD_DESC "Release -O3 -march=native" CACHE STRING "build description in run manifest")

find_package(Threads REQUIRED)

set(QBAM_SOURCES
  cpp/common/util.cpp
  cpp/common/json.cpp
  cpp/common/output.cpp
  cpp/io/events.cpp
  cpp/core/live_store.cpp
  cpp/core/router.cpp
  cpp/core/shard_index.cpp
  cpp/eval/cpu_meter.cpp
  cpp/eval/quality_attribution.cpp
  cpp/eval/exact_gt_check.cpp
  cpp/replay/replay_engine.cpp
  cpp/replay/cf_runner.cpp
  cpp/replay/policy_forecast.cpp
)

add_library(qbam_core STATIC ${QBAM_SOURCES})
target_include_directories(qbam_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${HNSWLIB_INCLUDE_DIR})
target_compile_options(qbam_core PUBLIC -O3 -march=${QBAM_MARCH} -Wall)
target_compile_definitions(qbam_core PUBLIC
  QBAM_GIT_SHA="${QBAM_GIT_SHA}"
  QBAM_BUILD_DESC="${QBAM_BUILD_DESC}")
target_link_libraries(qbam_core PUBLIC Threads::Threads)

add_executable(qbam_replay cpp/cli/replay_main.cpp)
target_link_libraries(qbam_replay PRIVATE qbam_core)

option(QBAM_BUILD_TESTS "Build engine unit tests" ON)
set(QBAM_TESTS
  test_crud
  test_refresh
  test_overfetch
  test_replay_determinism
  test_quality_attribution
  test_state_hash
  test_cf_branches
  test_budget_policies)

if(QBAM_BUILD_TESTS)
  enable_testing()
  foreach(test_name IN LISTS QBAM_TESTS)
    add_executable(${test_name} cpp/tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE qbam_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
