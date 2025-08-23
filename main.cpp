#include <iostream>
#include <helper.hpp>
#include <src.hpp>
#include <omp.h>

#define GAMMA 267522187

int main() {
    omp_set_num_threads(25);

    SimulationConfig cfg = loadConfig("../config.json");

    Voxel voxel(cfg.n, cfg.L, cfg.Xtot, cfg.eta, cfg.numberofprotons, cfg.B0vec, cfg.D, cfg.dt, cfg.mu, cfg.sigma, cfg.ratio);

    cfg.R2p =   GAMMA * cfg.B0 * cfg.Xtot / (9.0 * std::sqrt(3.0));
    cfg.tc = 6.0 * M_PI / (GAMMA * cfg.B0 * cfg.Xtot / cfg.eta);
    cfg.N = voxel.N_;

    std::cout << "Volume Fraction (eta): " << cfg.eta << std::endl;
    std::cout << "Voxel Susceptibility (Xtot): " << cfg.Xtot << std::endl;
    std::cout << "Occupieed Positions: " << voxel.Occupied_positions_.size() << std::endl;
    std::cout << "Sources Ratio: " << cfg.ratio << std::endl;
    std::cout << "R2': " << cfg.R2p  << std::endl;


    std::vector<double> times, signal, staticmag, star, Hyper, kappa2, kappa4, SEsignal;
    std::vector<std::vector<double>> allMoments, allCumulants;
    
    for (int t = 0; t < cfg.tsteps; ++t) {
        double current_time = t * cfg.dt;

        SignalResults result = voxel.SimulateDiffusionSteps(cfg.DiffSteps);
        std::complex<double> staticSignal = voxel.ComputeSignalStatic(current_time);
        
        
        double intermag = std::abs(result.totalPhase) * std::exp(-cfg.R2 * current_time);
        double analyticResult = std::exp(-cfg.R2p * current_time) * std::exp(-cfg.R2 * current_time);
        double hyperResult = std::exp(-cfg.eta * f(1.0 / cfg.tc, current_time)) * std::exp(-cfg.R2 * current_time);
        double k2 = std::abs(result.signal_kappa_2) * std::exp(-cfg.R2 * current_time);
        double k4 = std::abs(result.signal_kappa_4) * std::exp(-cfg.R2 * current_time);
        double statmag = std::abs(staticSignal) * std::exp(-cfg.R2 * current_time);
        double statp = std::arg(staticSignal);


        // Speichern
        times.push_back(current_time);
        signal.push_back(intermag);
        staticmag.push_back(statmag);
        star.push_back(analyticResult);
        Hyper.push_back(hyperResult);
        allMoments.push_back(result.moments);
        allCumulants.push_back(result.cumulants);
        kappa2.push_back(k2);
        kappa4.push_back(k4);
        SEsignal.push_back(0.0); // Placeholder for Spin Echo signal
      
        std::cout << "Timestep: " << current_time
                  << " Diffusion: " << intermag
                  << " Kappa_2: " << k2 << " kappa_4: " << k4
                  << " Static Dephasing: " << statmag
                  << " Hyper: " << hyperResult
                  << " Analytic: " << analyticResult
                  << std::endl;

    }

    std::cout << "Calculate Correlation function..." << std::endl; 
    auto correlation = voxel.ComputeTemporalACF(cfg.tsteps);

    std::cout << "Calculate Spacial Correlations..." << std::endl;
    auto [Cr_pp, Cr_nn, Cr_pn] = voxel.ComputeSpatialCorrelations();


    /*
    std::cout << "Start Simulation of SE..." << std::endl;
    for (int t = 0; t < cfg.tsteps; ++t) {
        double current_time = t * cfg.dt;

        std::complex<double> se = voxel.SimulateSpinEchoSignal(cfg.DiffSteps, cfg.dt, current_time);

        double seMag = std::abs(se) * std::exp(-cfg.R2 * current_time);

        SEsignal.push_back(seMag);

        std::cout << "SE Time: " << current_time << " SE Signal Magnitude: " << seMag << std::endl;
    }
    */

    saveConfigToJson(cfg, "output/config" + std::to_string(cfg.index) + ".json");
    std::string filename = "output/simulation_" + std::to_string(cfg.index) + ".nc";

    SaveAllToNetCDF(times, signal, staticmag, star, Hyper, correlation,
                    allMoments, allCumulants, kappa2, kappa4, SEsignal,
                    Cr_pp, Cr_nn, Cr_pn, filename, voxel.Protons_);
    
    SaveMapsToNETCDF("output/Maps_" + std::to_string(cfg.index) + ".nc", voxel.ChiMap_, voxel.Bz_, cfg.L);

    std::cout << "Simulation abgeschlossen und gespeichert unter: " << filename << std::endl;


    return 0;
}
