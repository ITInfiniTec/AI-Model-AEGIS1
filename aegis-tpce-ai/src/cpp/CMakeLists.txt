cmake_minimum_required(VERSION 3.10)

# Set the project name
project(aegis-tpce-ai-cpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(main src/main.cpp)

# Add include directories
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
