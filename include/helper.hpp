#ifndef HELPER_HPP
#define HELPER_HPP

#include <fftw3.h>
#include <complex>
#include <iostream>
#include <vector>
#include <string>

#include "src.hpp"


std::vector<std::vector<std::vector<std::complex<double>>>> applyFFT3D(const std::vector<std::vector<std::vector<double>>>& rSpace, int n);

std::vector<std::vector<std::vector<double>>> applyIFFT3D(const std::vector<std::vector<std::vector<std::complex<double>>>>& kSpace, int n, double B0_val); 

void shift3DArray(std::vector<std::vector<std::vector<double>>>& array, int n);

bool isElement(std::vector<std::vector<int>>& outer, std::vector<int>& inner);

void SaveSignalDecay(const std::vector<double>& times, const std::vector<double>& magnitudes, const std::vector<double>& signal, const std::vector<double>& star, std::string filename);

void SaveBzMap(const Voxel& voxel, const  std::string& filename);

void SaveDiffusionPaths(const Voxel& voxel, const std::string& filename);

void SaveChiMap(const Voxel& voxel, const std::string& filename);

#endif
