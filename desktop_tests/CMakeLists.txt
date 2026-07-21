cmake_minimum_required(VERSION 3.14)
project(NNAwareBLE_DesktopTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  json
  URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Populate(json)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE ${json_SOURCE_DIR})
# The header must land at <dir>/nlohmann/json.hpp for #include <nlohmann/json.hpp>
# to resolve -- FetchContent downloads a flat file, so place it correctly:
file(MAKE_DIRECTORY ${json_SOURCE_DIR}/nlohmann)
file(RENAME ${json_SOURCE_DIR}/json.hpp ${json_SOURCE_DIR}/nlohmann/json.hpp)

add_executable(protocol_compatibility protocol_compatibility.cpp)

# NNPacket.h / NNAddress.h live in src/, one directory up from desktop_tests/
target_include_directories(protocol_compatibility PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(protocol_compatibility PRIVATE nlohmann_json::nlohmann_json)

# Reasonably strict warnings -- this executable exists to catch real
# discrepancies, so it should not be silently tolerant of its own bugs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(protocol_compatibility PRIVATE -Wall -Wextra)
endif()

# Statically link the MinGW runtime on Windows so this exe doesn't depend on
# libstdc++-6.dll/libgcc_s_seh-1.dll/libwinpthread-1.dll being on PATH --
# pytest (via subprocess.run) inherits the caller's PATH, which won't
# include the MSYS2 toolchain bin directory outside of an MSYS2 shell,
# causing STATUS_DLL_NOT_FOUND (exit code 3221225781) at launch.
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(protocol_compatibility PRIVATE -static)
endif()

# ============================================================================
# Desktop unit/integration tests -- one executable per source header,
# matching the existing test_address.cpp / test_buffer.cpp naming
# convention. None of these need nlohmann_json (that's only needed for
# protocol_compatibility's JSON stdin/stdout interface) -- they link
# directly against the real headers in src/ and assert their own results.
# ============================================================================

set(NNAWARE_TEST_SOURCES
    test_address
    test_buffer
    test_node
    test_node_failure
    test_packet
    test_scheduler
    test_transport
    test_reference_network
    test_cross_layer_collision
)

enable_testing()

foreach(test_name ${NNAWARE_TEST_SOURCES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp)
        add_executable(${test_name} ${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${test_name} PRIVATE -Wall -Wextra)
        endif()
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(${test_name} PRIVATE -static)
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})
    else()
        message(STATUS "Skipping ${test_name}.cpp -- file not found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endforeach()