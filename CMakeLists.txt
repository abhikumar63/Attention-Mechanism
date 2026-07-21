cmake_minimum_required(VERSION 3.20)
project(AttentionEngine VERSION 1.0 LANGUAGES CXX)

# Enforced modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Forced Apple Silicon (ARM64) architecture
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Added aggressive optimization flags for native hardware AND opt-in to modern Accelerate headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=apple-m1 -DACCELERATE_NEW_LAPACK -g -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

# Locate Apple's Accelerate Framework for high-performance matrix math
find_library(ACCELERATE_FRAMEWORK Accelerate)
if(NOT ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR "Accelerate framework not found. Ensure you are building on macOS.")
endif()

add_executable(attention_engine src/main.cpp)
target_include_directories(attention_engine PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(attention_engine PRIVATE ${ACCELERATE_FRAMEWORK})