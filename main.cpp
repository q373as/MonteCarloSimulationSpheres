#include <iostream>
#include <helper.hpp>
#include <src.hpp>
#include <omp.h>

#define GAMMA 267522187

int main() {
    omp_set_num_threads(30); 

    SimulationConfig cfg = loadConfig("../config.json");

    double nd = static_cast<double>(cfg.n);
    double Vm =  1 / (nd * nd * nd);
    

    Voxel voxel(cfg.n, cfg.L, cfg.SIZE_arti, cfg.Xtot, cfg.eta, cfg.numberofprotons, cfg.B0vec, cfg.D, cfg.dt);
    

    cfg.R2p = (8 * M_PI*M_PI) * cfg.eta * GAMMA * 3 * cfg.Xtot / (9 * std::sqrt(3));

    std::cout << "Volume Fraction: " << cfg.eta  << std::endl;
    std::cout << "Voxel Suszept: " << cfg.Xtot << std::endl;
    std::cout << "Occupied Positions: " << voxel.Occupied_positions_.size()<< std::endl;
    std::cout << "Artifact Suszept: " << voxel.DeltaChi_ << std::endl;

    std::cout << "R2p " << cfg.R2p << std::endl;

    std::vector<double> magnitudes;
    std::vector<double> times;
    std::vector<double> signal;
    std::vector<double> star, Hyper;

    double inter,stat,result;
    cfg.tc = 3 / (4 * M_PI * GAMMA * cfg.B0 * cfg.Xtot);
        
    for(int t = 0; t < cfg.tsteps; t++){
        inter = std::abs(voxel.SimulateDiffusionSteps(cfg.DiffSteps, t * cfg.dt));
        stat = std::abs(voxel.ComputeSignalStatic(t * cfg.dt));

        result =  std::exp(- cfg.eta * f(1/cfg.tc, t * cfg.dt));
        std::cout << "Timestep: " << t*cfg.dt << "  Diffusion: " << inter << " Static Dephasing: " << stat << "  Hyper: " << result << "  Analytic: " << std::exp(- cfg.R2p * t * cfg.dt) <<  std::endl;

        signal.push_back(inter);
        magnitudes.push_back(stat);
        times.push_back(t * cfg.dt);
        Hyper.push_back(result);
        star.push_back(std::exp(- cfg.R2p * t * cfg.dt));
    }


    SaveSignalDecay(times, magnitudes, signal, star, Hyper, "output/signal_decay" + std::to_string(cfg.index) + ".txt");
    SaveBzMap(voxel, "output/Bz_map" + std::to_string(cfg.index) + ".txt");
    SaveChiMap(voxel, "output/ChiMap" + std::to_string(cfg.index) + ".txt");
    SaveDiffusionPaths(voxel, "output/diffusion_paths" + std::to_string(cfg.index) + ".txt");
    saveConfigToJson(cfg,"output/config" + std::to_string(cfg.index) + ".json");
    

    return 0;
}
