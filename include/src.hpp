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
    std::vector<double> position_;
    std::complex<double> phase_;
    std::vector<std::vector<double>> TrackPostitions_;
    std::vector<std::complex<double>> TrackPhases;

    Proton(const std::vector<double>& position);
};

// =====================
// ==== Voxel ==========
// =====================

class Voxel {
public:
    int n_;
    double L_;
    int N_ = 0;
    double SIZE_arti_;
    double Xtot_;
    int numberofprotons_;
    
    double dt_;
    double D_;
    double eta_;
    double DeltaChi_ = eta_ * Xtot_;
    
    double nb_ = static_cast<double>(n_);
   
    double Vm_ = 1 / (nb_ * nb_ * nb_);
    
    std::vector<double> B0_;
    std::vector<std::vector<std::vector<double>>> dz_;
    std::vector<std::vector<std::vector<double>>> ChiMap_;
    std::vector<std::vector<std::vector<double>>> Bz_;
    std::vector<std::vector<double>> ALL_positions_;
    std::set<std::vector<int>> Occupied_positions_;
    std::vector<Artifact> artifacts;
    std::vector<Proton> Protons_;
    std::vector<Proton> Protons_init_;

    fftw_complex* dz_fftw_;
    fftw_complex* ChiMap_fftw_;
    fftw_complex* Bz_fftw_;

    double B0_val_ = sqrt(B0_[0]*B0_[0] + B0_[1]*B0_[1] + B0_[2]*B0_[2]);
    
    Voxel(int n, double L, double SIZE_arti, double Xtot, double eta, int numberofprotons, std::vector<double> B0, double D, double dt);

    void SaveEveryEntry(std::vector<std::vector<std::vector<double>>> array);
    void CalculateDzMap();
    double interpolateBz(double x, double y, double z);
    std::complex<double> SimulateDiffusionSteps(int NrOfSteps, double t);
    std::complex<double> ComputeSignalStatic(double t);
};

std::set<std::vector<int>> GetOccupiedPositions(const std::vector<Artifact>& artifacts);

std::vector<std::vector<double>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, const std::set<std::vector<int>>& Occupied_positions);

#endif
