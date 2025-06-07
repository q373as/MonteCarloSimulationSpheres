#include <iostream>
#include <helper.hpp>
#include <src.hpp>
#include <omp.h>

#define GAMMA 267522187

int main() {
    omp_set_num_threads(25);

    SimulationConfig cfg = loadConfig("../config.json");

    double nd = static_cast<double>(cfg.n);
    double voxelVolume = 1.0 / (nd * nd * nd);

    Voxel voxel(cfg.n, cfg.L, cfg.Xtot, cfg.eta, cfg.numberofprotons, cfg.B0vec, cfg.D, cfg.dt, cfg.mu, cfg.sigma);


    cfg.R2p = (8.0 * M_PI * M_PI) * cfg.eta * GAMMA * 3.0 * cfg.Xtot / (9.0 * std::sqrt(3.0));

    std::cout << "Volume Fraction (eta): " << cfg.eta << std::endl;
    std::cout << "Voxel Susceptibility (Xtot): " << cfg.Xtot << std::endl;
    std::cout << "Occupieed Positions: " << voxel.Occupied_positions_.size() << std::endl;
    std::cout << "Artifact Susceptibility (DeltaChi): " << voxel.DeltaChi_ << std::endl;
    std::cout << "R2': " << cfg.R2p << std::endl;

    cfg.tc = 3.0 / (4.0 * M_PI * GAMMA * cfg.B0 * cfg.Xtot);
    cfg.N = voxel.N_;

    // Vektoren für die Speicherung
    std::vector<double> times, magnitudes, signal, star, Hyper, correlation, interPhase, statPhase;

    // Neu: für kappa-Signale (komplex) abs und phase speichern
    std::vector<double> kappa2Mag, kappa2Phase;
    std::vector<double> kappa4Mag, kappa4Phase;

    // Für Momente und Kumulanten (je 4 Einträge pro Zeitschritt)
    std::vector<std::vector<double>> allMoments;   // size: tsteps x 4
    std::vector<std::vector<double>> allCumulants; // size: tsteps x 4

    for (int t = 0; t < cfg.tsteps; ++t) {
        double current_time = t * cfg.dt;

        SignalResults result = voxel.SimulateDiffusionSteps(cfg.DiffSteps, current_time);

        // Diffusionssignal = totalPhase
        std::complex<double> interSignal = result.totalPhase;
        double intermag = std::abs(interSignal) * std::exp(-cfg.R2 * current_time);
        double interp = std::arg(interSignal);

        // kappa_2 Signal
        double k2mag = std::abs(result.signal_kappa_2) * std::exp(-cfg.R2 * current_time);
        double k2phase = std::arg(result.signal_kappa_2);

        // kappa_4 Signal
        double k4mag = std::abs(result.signal_kappa_4) * std::exp(-cfg.R2 * current_time);
        double k4phase = std::arg(result.signal_kappa_4);

        // Statisches Signal
        std::complex<double> staticSignal = voxel.ComputeSignalStatic(current_time);
        double statmag = std::abs(staticSignal) * std::exp(-cfg.R2 * current_time);
        double statp = std::arg(staticSignal);

        // Analytische Signale
        double hyperResult = std::exp(-cfg.eta * f(1.0 / cfg.tc, current_time)) * std::exp(-cfg.R2 * current_time);
        double analyticResult = std::exp(-cfg.R2p * current_time) * std::exp(-cfg.R2 * current_time);

        std::cout << "Timestep: " << current_time
                  << " Diffusion: " << intermag
                  << " Kappa_2: " << k2mag << " kappa_4: " << k4mag 
                  << " Static Dephasing: " << statmag
                  << " Hyper: " << hyperResult
                  << " Analytic: " << analyticResult
                  << std::endl;

        // Speichern
        times.push_back(current_time);
        magnitudes.push_back(statmag);
        signal.push_back(intermag);
        star.push_back(analyticResult);
        Hyper.push_back(hyperResult);
        interPhase.push_back(interp);
        statPhase.push_back(statp);

        kappa2Mag.push_back(k2mag);
        kappa2Phase.push_back(k2phase);
        kappa4Mag.push_back(k4mag);
        kappa4Phase.push_back(k4phase);

        // Momente und Kumulanten
        allMoments.push_back(result.moments);
        allCumulants.push_back(result.cumulants);
    }

    correlation = voxel.ComputeTemporalACF(cfg.tsteps);

    saveConfigToJson(cfg, "output/config" + std::to_string(cfg.index) + ".json");

    std::string filename = "output/simulation_" + std::to_string(cfg.index) + ".nc";

    std::vector<std::vector<std::vector<double>>> diffusionPaths;
    diffusionPaths.reserve(voxel.Protons_.size());

    for (const auto& proton : voxel.Protons_) {
        std::vector<std::vector<double>> protonPath;
        protonPath.reserve(proton.TrackPostitions_.size());

        for (const auto& pos : proton.TrackPostitions_) {
            protonPath.push_back({pos[0], pos[1], pos[2]});
        }
        diffusionPaths.push_back(std::move(protonPath));
    }

    // Speichere alle Daten, inkl. neuer kappa-Signale, Momente, Kumulanten
    SaveAllToNetCDF(times, magnitudes, signal, star, Hyper, correlation, interPhase, statPhase,
                    kappa2Mag, kappa2Phase, kappa4Mag, kappa4Phase,
                    allMoments, allCumulants,
                    filename, voxel.Bz_, voxel.ChiMap_, diffusionPaths);

    return 0;

}