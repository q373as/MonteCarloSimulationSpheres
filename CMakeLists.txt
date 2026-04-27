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

# NIfTI
find_path(NIFTI_INCLUDE_DIR nifti1_io.h PATHS /usr/include /usr/include/nifti)
find_library(NIFTI_LIB niftiio PATHS /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(ZNZ_LIB znz PATHS /usr/lib /usr/lib/x86_64-linux-gnu)

# ZLIB
find_package(ZLIB REQUIRED)

message(STATUS "Found NIfTI: ${NIFTI_LIB}")
include_directories(${NIFTI_INCLUDE_DIR})

# Conda Pfade (NetCDF)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})
include_directories(${CONDA_PREFIX}/include)
link_directories(${CONDA_PREFIX}/lib)

# --- NetCDF ---
find_path(NETCDF_INCLUDE_DIR netcdf PATHS /usr/include /usr/local/include)
find_library(NETCDF_CXX_LIB netcdf_c++4 PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NETCDF_C_LIB netcdf PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

include_directories(${NETCDF_INCLUDE_DIR})

# ---- Executable ----
add_executable(simulation ${SOURCES})

# ---- Libraries ----
target_link_libraries(simulation
    PRIVATE
    ${FFTW3_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${NETCDF_CXX_LIB}
    ${NETCDF_C_LIB}
    ${NIFTI_LIB}
    ${ZNZ_LIB}
    ZLIB::ZLIB
    nlohmann_json::nlohmann_json
)

