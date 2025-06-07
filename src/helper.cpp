#include <iostream>
#include <string>
#include <fstream>
#include "helper.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;


SimulationConfig loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filename);
    }

    json config_json;
    file >> config_json;

    SimulationConfig config;
    config.n = config_json.value("n", 500);
    config.numberofprotons = config_json.value("numberofprotons", 100000);
    config.L = config_json.value("L", 1.0);
    config.mu = config_json.value("mu", 10.0);
    config.sigma = config_json.value("sigma", 10.0);
    config.eta = config_json.value("eta", 0.01);
    config.Xtot = config_json.value("Xtot", 1e-7);
    config.dt = config_json.value("dt", 0.0001);
    config.D = config_json.value("D", 5e-3);
    config.B0vec = config_json.value("B0", std::vector<double>{0.0, 0.0, 3.0});
    config.B0 = sqrt(config.B0vec[0] * config.B0vec[0] + config.B0vec[1] * config.B0vec[1] + config.B0vec[2] * config.B0vec[2]);
    config.index = config_json.value("index", 0);
    config.DiffSteps = config_json.value("DiffSteps", 10.0);
    config.tsteps = config_json.value("tsteps", 1000);
    config.R2 = config_json.value("R2", 10);
    return config;
}

std::vector<std::vector<std::vector<double>>> applyIFFT3D(const std::vector<std::vector<std::vector<std::complex<double>>>>& kSpace, int n, double B0_val) {
    std::vector<std::vector<std::vector<double>>> rSpace(n, std::vector<std::vector<double>>(n, std::vector<double>(n)));

 
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);



    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                in[i * n * n + j * n + k][0] = kSpace[i][j][k].real();  // Realteil
                in[i * n * n + j * n + k][1] = kSpace[i][j][k].imag();  // Imaginärteil
            }

     
    fftw_plan plan = fftw_plan_dft_3d(n, n, n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                double shift = (out[i * n * n + j * n + k][0]);
                rSpace[i][j][k] = B0_val * shift / (n * n * n);  
            }

  
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return rSpace; 
}

std::vector<std::vector<std::vector<std::complex<double>>>> applyFFT3D(const std::vector<std::vector<std::vector<double>>>& rSpace, int n) {
    std::vector<std::vector<std::vector<std::complex<double>>>> kSpace(n, std::vector<std::vector<std::complex<double>>>(n, std::vector<std::complex<double>>(n)));

    
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);

    for (int i = 0; i < n * n * n; ++i) {
        in[i][0] = 0.0;  
        in[i][1] = 0.0;  
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                in[i * n * n + j * n + k][0] = rSpace[i][j][k]; 
                in[i * n * n + j * n + k][1] = 0.0;  

            }

    
 
    fftw_plan plan = fftw_plan_dft_3d(n, n, n, in, out, FFTW_FORWARD, FFTW_MEASURE);

    fftw_execute(plan);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                kSpace[i][j][k] = std::complex<double>(out[i * n * n + j * n + k][0], out[i * n * n + j * n + k][1]);
            }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return kSpace;
}

void shift3DArray(std::vector<std::vector<std::vector<double>>>& array, int n) {
    std::vector<std::vector<std::vector<double>>> shiftedArray(n,
        std::vector<std::vector<double>>(n,
        std::vector<double>(n)));

    // Verschiebe das Array um n/2 in jeder Dimension (periodisch)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                // Berechne die neuen Indizes mit modulo n (periodische Verschiebung)
                int new_i = (i + n / 2) % n;
                int new_j = (j + n / 2) % n;
                int new_k = (k + n / 2) % n;

                // Verschiebe den Wert in das neue Array
                shiftedArray[new_i][new_j][new_k] = array[i][j][k];
            }
        }
    }

    // Setze das verschobene Array zurück ins Originalarray
    array = shiftedArray;
}

bool isElement(const std::set<std::vector<int>>& s, const std::vector<double>& v) {
    std::vector<int> v_rounded;
    v_rounded.reserve(v.size());
    for (double val : v) {
        v_rounded.push_back(static_cast<int>(std::round(val)));
    }
    return s.find(v_rounded) != s.end();
}

double pochhammer(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i)
        result *= (x + i);
    return result;
}

double hypergeom_1F2(double a, double b1, double b2, double z, int max_terms, double tol) {
    double sum = 1.0;
    double term = 1.0;
    for (int n = 1; n < max_terms; ++n) {
        term *= (a + n - 1) * z / ((b1 + n - 1) * (b2 + n - 1) * n);
        sum += term;
        //std::cout << "Term " << n << ": " << term << std::endl;
        if (std::abs(term) < tol) break;
    }
    return sum;
}

double f(double delta_omega, double t) {
    double x = delta_omega * t;
    double z = -9.0 / 16.0 * x * x;
    return hypergeom_1F2(-0.5, 0.75, 1.25, z) - 1.0;
}

void saveConfigToJson(const SimulationConfig& cfg, const std::string& filename) {
    json j;
    j["n"] = cfg.n;
    j["numberofprotons"] = cfg.numberofprotons;
    j["L"] = cfg.L;
    j["mu"] = cfg.mu;
    j["sigma"] = cfg.sigma;
    j["DeltaChi"] = cfg.DeltaChi;
    j["N"] = cfg.N;
    j["dt"] = cfg.dt;
    j["D"] = cfg.D;
    j["B0vec"] = cfg.B0vec;
    j["B0"] = cfg.B0;
    j["index"] = cfg.index;
    j["DiffSteps"] = cfg.DiffSteps;
    j["tsteps"] = cfg.tsteps;
    j["eta"] = cfg.eta;
    j["Xtot"] = cfg.Xtot;
    j["tc"] = cfg.tc;
    j["R2p"] = cfg.R2p;

    std::ofstream file(filename);
    file << j.dump(2);  // pretty print with indent=2
}

void SaveAllToNetCDF(
    const std::vector<double>& times,
    const std::vector<double>& magnitudes,
    const std::vector<double>& signal,
    const std::vector<double>& star,
    const std::vector<double>& Hyper,
    const std::vector<double>& correlation,
    const std::vector<double>& interPhase,
    const std::vector<double>& statPhase,
    const std::vector<double>& kappa2Mag,
    const std::vector<double>& kappa2Phase,
    const std::vector<double>& kappa4Mag,
    const std::vector<double>& kappa4Phase,
    const std::vector<std::vector<double>>& allMoments,
    const std::vector<std::vector<double>>& allCumulants,
    const std::string& filename,
    const std::vector<std::vector<std::vector<double>>>& Bz,
    const std::vector<std::vector<std::vector<double>>>& ChiMap,
    const std::vector<std::vector<std::vector<double>>>& diffusionPaths
) {
    try {
        NcFile dataFile(filename, NcFile::replace);

        // Dimensionen definieren
        auto tsteps = times.size();
        auto moments_size = 4;
        auto nProtons = diffusionPaths.size();

        // Zeit-Dimension
        NcDim timeDim = dataFile.addDim("time", tsteps);

        // Moments und Cumulants Dimension
        NcDim momentsDim = dataFile.addDim("moments", moments_size);

        // Diffusionspfade Dimensionen: protonen, steps, coords (3)
        size_t maxSteps = 0;
        for (const auto& path : diffusionPaths)
            if (path.size() > maxSteps)
                maxSteps = path.size();

        NcDim protonDim = dataFile.addDim("protons", nProtons);
        NcDim stepDim = dataFile.addDim("steps", maxSteps);
        NcDim coordDim = dataFile.addDim("coord", 3);

        // Variablen anlegen
        auto timeVar = dataFile.addVar("time", ncDouble, timeDim);
        auto magnVar = dataFile.addVar("magnitude_static", ncDouble, timeDim);
        auto signalVar = dataFile.addVar("signal_diffusion", ncDouble, timeDim);
        auto starVar = dataFile.addVar("analytic_star", ncDouble, timeDim);
        auto hyperVar = dataFile.addVar("analytic_hyper", ncDouble, timeDim);
        auto corrVar = dataFile.addVar("correlation", ncDouble, timeDim);
        auto interPhaseVar = dataFile.addVar("interPhase", ncDouble, timeDim);
        auto statPhaseVar = dataFile.addVar("statPhase", ncDouble, timeDim);

        auto k2MagVar = dataFile.addVar("kappa2_magnitude", ncDouble, timeDim);
        auto k2PhaseVar = dataFile.addVar("kappa2_phase", ncDouble, timeDim);
        auto k4MagVar = dataFile.addVar("kappa4_magnitude", ncDouble, timeDim);
        auto k4PhaseVar = dataFile.addVar("kappa4_phase", ncDouble, timeDim);

        // Moments und Cumulants (2D: time x moments)
        std::vector<NcDim> dims2D = {timeDim, momentsDim};
        auto momentsVar = dataFile.addVar("moments", ncDouble, dims2D);
        auto cumulantsVar = dataFile.addVar("cumulants", ncDouble, dims2D);

        // 3D Voxel-Felder: Bz, ChiMap
        size_t Nx = Bz.size();
        size_t Ny = (Nx > 0) ? Bz[0].size() : 0;
        size_t Nz = (Ny > 0) ? Bz[0][0].size() : 0;

        NcDim xDim = dataFile.addDim("x", Nx);
        NcDim yDim = dataFile.addDim("y", Ny);
        NcDim zDim = dataFile.addDim("z", Nz);

        auto BzVar = dataFile.addVar("Bz", ncDouble, {xDim, yDim, zDim});
        auto ChiVar = dataFile.addVar("ChiMap", ncDouble, {xDim, yDim, zDim});

        // Diffusionspfade: protons x steps x 3 coords
        auto diffusionVar = dataFile.addVar("diffusionPaths", ncDouble, {protonDim, stepDim, coordDim});

        // Daten schreiben

        timeVar.putVar(times.data());
        magnVar.putVar(magnitudes.data());
        signalVar.putVar(signal.data());
        starVar.putVar(star.data());
        hyperVar.putVar(Hyper.data());
        corrVar.putVar(correlation.data());
        interPhaseVar.putVar(interPhase.data());
        statPhaseVar.putVar(statPhase.data());

        k2MagVar.putVar(kappa2Mag.data());
        k2PhaseVar.putVar(kappa2Phase.data());
        k4MagVar.putVar(kappa4Mag.data());
        k4PhaseVar.putVar(kappa4Phase.data());

        // Momente & Kumulanten müssen als flaches Array geschrieben werden
        std::vector<double> momentsFlat(tsteps * moments_size);
        std::vector<double> cumulantsFlat(tsteps * moments_size);

        for (size_t i = 0; i < tsteps; ++i) {
            for (size_t j = 0; j < moments_size; ++j) {
                momentsFlat[i * moments_size + j] = allMoments[i][j];
                cumulantsFlat[i * moments_size + j] = allCumulants[i][j];
            }
        }

        momentsVar.putVar(momentsFlat.data());
        cumulantsVar.putVar(cumulantsFlat.data());

        // Bz und ChiMap (3D Felder)
        std::vector<double> BzFlat;
        BzFlat.reserve(Nx * Ny * Nz);
        for (size_t i = 0; i < Nx; ++i)
            for (size_t j = 0; j < Ny; ++j)
                for (size_t k = 0; k < Nz; ++k)
                    BzFlat.push_back(Bz[i][j][k]);
        BzVar.putVar(BzFlat.data());

        std::vector<double> ChiFlat;
        ChiFlat.reserve(Nx * Ny * Nz);
        for (size_t i = 0; i < Nx; ++i)
            for (size_t j = 0; j < Ny; ++j)
                for (size_t k = 0; k < Nz; ++k)
                    ChiFlat.push_back(ChiMap[i][j][k]);
        ChiVar.putVar(ChiFlat.data());

        // Diffusionspfade (protons x steps x 3)
        // Nullwerte bei kürzeren Pfaden auffüllen mit NaN oder 0 (hier 0)
        std::vector<double> diffusionFlat(nProtons * maxSteps * 3, 0.0);
        for (size_t p = 0; p < nProtons; ++p) {
            for (size_t s = 0; s < diffusionPaths[p].size(); ++s) {
                diffusionFlat[(p * maxSteps + s) * 3 + 0] = diffusionPaths[p][s][0];
                diffusionFlat[(p * maxSteps + s) * 3 + 1] = diffusionPaths[p][s][1];
                diffusionFlat[(p * maxSteps + s) * 3 + 2] = diffusionPaths[p][s][2];
            }
        }
        diffusionVar.putVar(diffusionFlat.data());

        std::cout << "Daten erfolgreich in NetCDF Datei gespeichert: " << filename << std::endl;

    } catch (NcException& e) {
        std::cerr << "Fehler beim Schreiben der NetCDF Datei: " << e.what() << std::endl;
    }
}