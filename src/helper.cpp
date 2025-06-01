#include <iostream>
#include <string>
#include <fstream>
#include "helper.hpp"



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

void SaveSignalDecay(const std::vector<double>& times, const std::vector<double>& magnitudes, const std::vector<double>& signal, const std::vector<double>& star, const std::vector<double>& Hyper, std::string filename){
        std::ofstream outFile(filename);

        if (!outFile.is_open()) {
        std::cerr << "Error opening signal_decay file for writing!\n";
        return;
        }

        outFile << "Time\tStaticDephasing\tDiffusion\tAnalytic\tHyper\n";
        for (size_t i = 0; i < times.size(); ++i) {
                outFile << times[i] << "\t" 
                << magnitudes[i] << "\t" 
                << signal[i] << "\t" 
                << star[i] << "\t"
                << Hyper[i] << "\n";
                }
        std::cout << "Signal decay saved\n";
    }

void SaveBzMap(const Voxel& voxel, const std::string& filename){
    std::ofstream bzFile(filename);
    if (!bzFile.is_open()) {
        std::cerr << "Error opening Bz_map file for writing!\n";
        return;
    }

    for (size_t i = 0; i < voxel.Bz_.size(); ++i) {
        for (size_t j = 0; j < voxel.Bz_[i].size(); ++j) {
            for (size_t k = 0; k < voxel.Bz_[i][j].size(); ++k) {
                bzFile << i << " " << j << " " << k << " " << voxel.Bz_[i][j][k] << "\n";
            }
        }
    }
    std::cout << "Bz map saved" << std::endl;
}

void SaveDiffusionPaths(const Voxel& voxel, const std::string& filename) {
    std::ofstream pathsFile(filename);
    if (pathsFile.is_open()) {
        for (size_t p = 0; p < voxel.Protons_.size(); ++p) {
            pathsFile << "# Proton " << p << "\n";
            const auto& proton = voxel.Protons_[p];
            for (size_t step = 0; step < proton.TrackPostitions_.size(); ++step) {
                const auto& pos = proton.TrackPostitions_[step];
                pathsFile << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\n";
            }
            pathsFile << "\n";  // Leerzeile zwischen Protonen
        }
        std::cout << "Diffusion paths saved\n";
    } else {
        std::cerr << "Error opening diffusion_paths file for writing!\n";
    }
}

void SaveChiMap(const Voxel& voxel, const std::string& filename){
    std::ofstream chiFile(filename);
    if (!chiFile.is_open()) {
        std::cerr << "Error opening Chi_map file for writing!\n";
        return;
    }

    for (size_t i = 0; i < voxel.ChiMap_.size(); ++i) {
        for (size_t j = 0; j < voxel.ChiMap_[i].size(); ++j) {
            for (size_t k = 0; k < voxel.ChiMap_[i][j].size(); ++k) {
                chiFile << i << " " << j << " " << k << " " << voxel.ChiMap_[i][j][k] << "\n";
            }
        }
    }
    std::cout << "Chi map saved\n";
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