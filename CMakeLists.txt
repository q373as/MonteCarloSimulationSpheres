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

# Manuelles Hinzufügen von NetCDF aus Conda (Pfad an dein Conda env anpassen)
set(CONDA_PREFIX $ENV{CONDA_PREFIX})  # aktuelles Conda Environment

include_directories(${CONDA_PREFIX}/include)
link_directories(${CONDA_PREFIX}/lib)

add_executable(simulation ${SOURCES})

target_link_libraries(simulation
    PRIVATE
    ${FFTW3_LIBRARIES}
    OpenMP::OpenMP_CXX
    netcdf_c++4      # Name der NetCDF C++-Bibliothek in Conda (meist so)
    netcdf          # manchmal zusätzlich die C-Bibliothek verlinken
)

