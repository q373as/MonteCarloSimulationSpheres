cmake_minimum_required(VERSION 3.10)
project(MySimulationProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-Ordner einbinden
include_directories(include)

# Quellcode-Dateien einbinden
set(SOURCES
    main.cpp
    src/helper.cpp
    src/src.cpp
)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# FFTW
find_package(FFTW3 REQUIRED)
if(FFTW3_FOUND)
    message(STATUS "Found FFTW")
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "FFTW not found!")
endif()

# Executable bauen
add_executable(simulation ${SOURCES})

# Linker-Bibliotheken
target_link_libraries(simulation
    ${FFTW_LIBRARIES}
    -lfftw3
    OpenMP::OpenMP_CXX
)
