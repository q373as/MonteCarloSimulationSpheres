#include <iostream>
#include <helper.hpp>
#include <src.hpp>
#include <omp.h>

#define GAMMA 267522187

int main() {
    omp_set_num_threads(30); 

    SimulationConfig cfg = loadConfig("../config.json");

    double nd = static_cast<double>(cfg.n);
    double Vm = 1 / (nd * nd * nd);
   
    Voxel voxel(cfg.n, cfg.L, cfg.SIZE_arti, cfg.DeltaChi, cfg.N, cfg.numberofprotons, cfg.B0vec, cfg.D, cfg.dt);


    double eta = voxel.Occupied_positions_.size() * Vm;
    double xtot = voxel.Occupied_positions_.size() * cfg.DeltaChi;
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
        inter = std::abs(voxel.SimulateDiffusionSteps(10, t * cfg.dt));
        stat = std::abs(voxel.ComputeSignalStatic(t * cfg.dt));

        std::cout << "Timestep: " << t*cfg.dt << "  Diffusion: " << inter << " Static Dephasing: " << stat << "  Analytic: " << std::exp(- R2p * t * cfg.dt) << std::endl;

        signal.push_back(inter);
        magnitudes.push_back(stat);
        times.push_back(t * cfg.dt);
        star.push_back(std::exp(- R2p * t * cfg.dt));
    }

    SaveSignalDecay(times, magnitudes, signal, star, "output/signal_decay" + std::to_string(0) + ".txt");
    SaveBzMap(voxel, "output/Bz_map" + std::to_string(0) + ".txt");
    SaveChiMap(voxel, "output/ChiMap" + std::to_string(0) + ".txt");
    SaveDiffusionPaths(voxel, "output/diffusion_paths" + std::to_string(0) + ".txt");
      
    

    return 0;
}
