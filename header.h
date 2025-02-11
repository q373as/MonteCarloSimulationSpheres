#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <fftw3.h>
#include <omp.h>
#include <ctime>
#include <random>
#include <complex>



void plotBzWithParticles(std::vector<std::vector<std::vector<double>>> Bz, std::vector<std::vector<int>>& particles, int n);
std::vector<std::vector<std::vector<double>>> convertComplexToDouble(const std::vector<std::vector<std::vector<std::complex<double>>>>& complexVec);
void print3DVector(const std::vector<std::vector<std::vector<double>>>& vec);
bool isElement(std::vector<std::vector<int>>& outer, std::vector<int>& inner); 


std::vector<std::vector<int>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, std::vector<std::vector<int>> Occupied_positions);
std::vector<std::vector<int>> GetOccupiedPositions(std::vector<Artifact>& artifacts);

class Artifact {
    public:
        std::vector<int> positionmain_;
        double suscept_;
        int size_;
        std::vector<std::vector<int>> positions_;

        Artifact(std::vector<int> positionmain, int pm, int size, double suscept, int n);
};

class Voxel {
    public:
        int n_;
        double L_;
        int N_;
        int SIZE_arti_;
        double DeltaChi_;
        int numberofprotons_;

        std::vector<double> B0_;
        std::vector<std::vector<std::vector<double>>> dz_;
        std::vector<std::vector<std::vector<double>>> ChiMap_;
        std::vector<std::vector<std::vector<double>>> Bz_;
        std::vector<std::vector<int>> ALL_positions_;
        std::vector<std::vector<int>> Occupied_positions_;

        fftw_complex* dz_fftw_;
        fftw_complex* ChiMap_fftw_;
        fftw_complex* Bz_fftw_;

        Voxel(int n, double L, int SIZE_arti, double DeltaChi, int N, int numberofprotons, std::vector<double> B0);
      

    void CalculateDzMap();
    void SimulateDiffusionSteps(std::vector<double>& prevpos, int NrOfSteps);
    void Plot4D(const std::vector<double>& data, const std::string& label, const std::string& name);
};


std::vector<std::vector<std::vector<std::complex<double>>>> applyFFT3D(const std::vector<std::vector<std::vector<double>>>& rSpace, int n);
std::vector<std::vector<std::vector<double>>> applyIFFT3D(const std::vector<std::vector<std::vector<std::complex<double>>>>& kSpace, int n);

