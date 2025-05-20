#ifndef HELPER_HPP
#define HELPER_HPP

#include <fftw3.h>
#include <complex>
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include "src.hpp"


using json = nlohmann::json;

struct SimulationConfig {
    int n;
    int numberofprotons;
    double L;
    double SIZE_arti;
    double DeltaChi;
    int N;
    double dt;
    double D;
    std::vector<double> B0vec;
    double B0;
    int index;
    double DiffSteps; 
    int tsteps;
    double eta; 
    double Xtot; 
    double tc;
    double R2p;

};

SimulationConfig loadConfig(const std::string& filename);
    
std::vector<std::vector<std::vector<std::complex<double>>>> applyFFT3D(const std::vector<std::vector<std::vector<double>>>& rSpace, int n);

std::vector<std::vector<std::vector<double>>> applyIFFT3D(const std::vector<std::vector<std::vector<std::complex<double>>>>& kSpace, int n, double B0_val); 

void shift3DArray(std::vector<std::vector<std::vector<double>>>& array, int n);

bool isElement(const std::set<std::vector<int>>& s, const std::vector<double>& v);

void SaveSignalDecay(const std::vector<double>& times, const std::vector<double>& magnitudes, const std::vector<double>& signal, const std::vector<double>& star, const std::vector<double>& Hyper, std::string filename);

void SaveBzMap(const Voxel& voxel, const  std::string& filename);

void SaveDiffusionPaths(const Voxel& voxel, const std::string& filename);

void SaveChiMap(const Voxel& voxel, const std::string& filename);

// Pochhammer symbol (rising factorial)
double pochhammer(double x, int n);

double hypergeom_1F2(double a, double b1, double b2, double z, int max_terms = 100, double tol = 1e-50);

double f(double delta_omega, double t);

void saveConfigToJson(const SimulationConfig& cfg, const std::string& filename);

#endif
