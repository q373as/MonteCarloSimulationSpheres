#include <iostream>
#include <helper.hpp>
#include <src.hpp>
#include <omp.h>

#define GAMMA 267522187

int main() {
    omp_set_num_threads(30); 
    int n = 500;  //grid points 
    int numberofprotons = 1e5;
    double L = 1.0f; // voxelsize
    int SIZE_arti = 10; // sizeof artifact
    double DeltaChi =5e-15;
    
    
    int N = 2000; //numberfartifatcs
    double dt = 0.0001;
    double nd = static_cast<double>(n);
    double Vm = 1 / (nd * nd * nd);
    double D = 5e-3; // Diffusion coefficient

    for (int d = 5; d < 15; d++) {
        D = d * 1e-3;
        std::vector<double> B0 = {0.0f, 0.0f, 3.0f};  // External magnetic field
        Voxel voxel(n, L, SIZE_arti, DeltaChi, N, numberofprotons, B0, D, dt);


        double eta = voxel.Occupied_positions_.size() * Vm;
        double xtot = voxel.Occupied_positions_.size() * DeltaChi;
        double R2p = (8 * M_PI*M_PI) * eta * GAMMA * 3 * xtot / (9 * std::sqrt(3));

        std::cout << "Volume Fraction: " << eta  << std::endl;
        std::cout << "Voxel Suszept: " << xtot<< std::endl;
        std::cout << "R2p " << R2p << std::endl;
        std::vector<double> magnitudes;
        std::vector<double> times;
        std::vector<double> signal;
        std::vector<double> star;
        double inter,stat;

            
        for(int t = 0; t < 1000; t++){
            inter = std::abs(voxel.SimulateDiffusionSteps(10, t * dt));
            stat = std::abs(voxel.ComputeSignalStatic(t * dt));

            std::cout << "Timestep: " << t*dt << "  Diffusion: " << inter << " Static Dephasing: " << stat << "  Analytic: " << std::exp(- R2p * t * dt) << std::endl;

            signal.push_back(inter);
            magnitudes.push_back(stat);
            times.push_back(t * dt);
            star.push_back(std::exp(- R2p * t * dt));
        }

        SaveSignalDecay(times, magnitudes, signal, star, "output/signal_decay" + std::to_string(d) + ".txt");
        SaveBzMap(voxel, "output/Bz_map" + std::to_string(d) + ".txt");
        SaveChiMap(voxel, "output/ChiMap" + std::to_string(d) + ".txt");
        SaveDiffusionPaths(voxel, "output/diffusion_paths" + std::to_string(d) + ".txt");
      
    }

        
    return 0;
}
