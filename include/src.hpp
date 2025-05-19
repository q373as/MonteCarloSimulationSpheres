#ifndef SRC_HPP
#define SRC_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <set>

// =====================
// ==== Artifact =======
// =====================

class Artifact {
public:
    std::vector<int> positionmain_;
    double suscept_;
    double totsuscept_;
    int size_;
    std::vector<std::vector<int>> positions_;

    Artifact(std::vector<int> positionmain, int pm, int size, double suscept, int n);
};

// =====================
// ==== Proton =========
// =====================

class Proton {
public:
    std::vector<int> position_;
    std::complex<double> phase_;
    std::vector<std::vector<int>> TrackPostitions_;
    std::vector<std::complex<double>> TrackPhases;

    Proton(const std::vector<int>& position);
};

// =====================
// ==== Voxel ==========
// =====================

class Voxel {
public:
    int n_;
    double L_;
    int N_;
    int SIZE_arti_;
    double DeltaChi_;
    int numberofprotons_;
    double B0_val_;
    double dt_;
    double D_;

    double nb_ = static_cast<double>(n_);
   
    double Vm_ = 1 / (nb_ * nb_ * nb_);
    
    std::vector<double> B0_;
    std::vector<std::vector<std::vector<double>>> dz_;
    std::vector<std::vector<std::vector<double>>> ChiMap_;
    std::vector<std::vector<std::vector<double>>> Bz_;
    std::vector<std::vector<int>> ALL_positions_;
    std::set<std::vector<int>> Occupied_positions_;
    
    std::vector<Proton> Protons_;
    std::vector<Proton> Protons_init_;

    fftw_complex* dz_fftw_;
    fftw_complex* ChiMap_fftw_;
    fftw_complex* Bz_fftw_;

    Voxel(int n, double L, int SIZE_arti, double DeltaChi, int N, int numberofprotons, std::vector<double> B0, double D, double dt);

    void SaveEveryEntry(std::vector<std::vector<std::vector<double>>> array);
    void CalculateDzMap();
    std::complex<double> SimulateDiffusionSteps(int NrOfSteps, double t);
    std::complex<double> ComputeSignalStatic(double t);
};

std::set<std::vector<int>> GetOccupiedPositions(const std::vector<Artifact>& artifacts);

std::vector<std::vector<int>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, const std::set<std::vector<int>>& Occupied_positions);

#endif
