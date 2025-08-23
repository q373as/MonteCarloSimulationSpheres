cmake_minimum_required(VERSION 3.10)
project(MySimulationProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    main.cpp
    src/helper.cpp
    src/src.cpp
)

# OpenMP
find_package(OpenMP REQUIRED)

# FFTW3
find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})

# nlohmann_json
find_package(nlohmann_json REQUIRED)

# Conda Pfade (NetCDF)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})
include_directories(${CONDA_PREFIX}/include)
link_directories(${CONDA_PREFIX}/lib)

# ---- Executable zuerst definieren ----
add_executable(simulation ${SOURCES})

# ---- Dann Libraries anhängen ----
target_link_libraries(simulation
    PRIVATE
    ${FFTW3_LIBRARIES}
    OpenMP::OpenMP_CXX
    netcdf_c++4
    netcdf
    nlohmann_json::nlohmann_json
)

