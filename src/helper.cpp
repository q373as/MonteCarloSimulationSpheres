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
    config.R2 = config_json.value("R2", 20);
    config.ratio = config_json.value("ratio", 1.0);
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

void shift3DArray(std::vector<std::vector<std::vector<std::complex<double>>>>& array, int n) {
    std::vector<std::vector<std::vector<std::complex<double>>>> shiftedArray(
        n, std::vector<std::vector<std::complex<double>>>(
            n, std::vector<std::complex<double>>(n)));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int new_i = (i + n / 2) % n;
                int new_j = (j + n / 2) % n;
                int new_k = (k + n / 2) % n;

                shiftedArray[new_i][new_j][new_k] = array[i][j][k];
            }
        }
    }
    array.swap(shiftedArray);
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
    j["R2"] = cfg.R2;
    j["R2p"] = cfg.R2p;
    j["ratio"] = cfg.ratio;

    std::ofstream file(filename);
    file << j.dump(2);  // pretty print with indent=2
}

void SaveAllToNetCDF(
    const std::vector<double>& times,
    const std::vector<double>& signal,
    const std::vector<double>& staticmag,
    const std::vector<double>& star,
    const std::vector<double>& Hyper,
    const std::vector<double>& correlation,
    const std::vector<std::vector<double>>& allMoments,
    const std::vector<std::vector<double>>& allCumulants,
    const std::vector<double>& k2,
    const std::vector<double>& k4,
    const std::vector<double>& SEsignal,
    const std::vector<double>& Cr_pp,       // <--- NEU
    const std::vector<double>& Cr_nn,       // <--- NEU
    const std::vector<double>& Cr_pn,       // <--- NEU
    const std::string& filename,
    const std::vector<Proton>& protons)
{
    try {
        NcFile dataFile(filename, NcFile::replace);

        // Dimensionen
        auto dim_time = dataFile.addDim("time", times.size());
        auto dim_protons = dataFile.addDim("protons", protons.size());
        auto dim_moment_order = dataFile.addDim("moment_order", 4);

        // Hier neue Dimension für r-Bins
        auto dim_r = dataFile.addDim("r", Cr_pp.size());

        // Zeiten
        auto var_times = dataFile.addVar("time", ncDouble, dim_time);
        var_times.putVar(times.data());

        // Andere 1D-Variablen
        auto var_signal = dataFile.addVar("Diffusion", ncDouble, dim_time);
        var_signal.putVar(signal.data());
        
        auto var_static = dataFile.addVar("StaticDephasing", ncDouble, dim_time);
        var_static.putVar(staticmag.data());

        auto var_star = dataFile.addVar("Analytic", ncDouble, dim_time);
        var_star.putVar(star.data());

        auto var_hyper = dataFile.addVar("Hyper", ncDouble, dim_time);
        var_hyper.putVar(Hyper.data());

        auto var_correlation = dataFile.addVar("correlation", ncDouble, dim_time);
        var_correlation.putVar(correlation.data());

        auto var_SEsignal = dataFile.addVar("SEsignal", ncDouble, dim_time);
        var_SEsignal.putVar(SEsignal.data());

        // Momente
        auto var_moments = dataFile.addVar("moments", ncDouble, {dim_time, dim_moment_order});
        std::vector<double> moments_flat;
        moments_flat.reserve(times.size() * 4);
        for (const auto& m : allMoments)
            moments_flat.insert(moments_flat.end(), m.begin(), m.end());
        var_moments.putVar(moments_flat.data());

        // Kumulanten
        auto var_cumulants = dataFile.addVar("cumulants", ncDouble, {dim_time, dim_moment_order});
        std::vector<double> cumulants_flat;
        cumulants_flat.reserve(times.size() * 4);
        for (const auto& k : allCumulants)
            cumulants_flat.insert(cumulants_flat.end(), k.begin(), k.end());
        var_cumulants.putVar(cumulants_flat.data());

        // k2 (kappa2) hinzufügen
        auto var_k2 = dataFile.addVar("kappa2", ncDouble, dim_time);
        var_k2.putVar(k2.data());

        // k4 (kappa4) hinzufügen
        auto var_k4 = dataFile.addVar("kappa4", ncDouble, dim_time);
        var_k4.putVar(k4.data());

        // === NEU: Korrelationsfunktionen ===
        auto var_Cpp = dataFile.addVar("Correlation_pp", ncDouble, dim_r);
        var_Cpp.putVar(Cr_pp.data());

        auto var_Cnn = dataFile.addVar("Correlation_nn", ncDouble, dim_r);
        var_Cnn.putVar(Cr_nn.data());

        auto var_Cpn = dataFile.addVar("Correlation_pn", ncDouble, dim_r);
        var_Cpn.putVar(Cr_pn.data());

        
        /*
        // Protonenpositionen und Phasen
        size_t max_steps = 0;
        for (const auto& p : protons)
            if (p.TrackPostitions_.size() > max_steps)
                max_steps = p.TrackPostitions_.size();

        auto dim_steps = dataFile.addDim("steps", max_steps);
        auto dim_xyz = dataFile.addDim("xyz", 3);

        auto var_positions = dataFile.addVar("proton_positions", ncDouble, {dim_protons, dim_steps, dim_xyz});
        auto var_phases_real = dataFile.addVar("proton_phases_real", ncDouble, {dim_protons, dim_steps});
        auto var_phases_imag = dataFile.addVar("proton_phases_imag", ncDouble, {dim_protons, dim_steps});

        std::vector<double> positions_data(protons.size() * max_steps * 3, 0.0);
        std::vector<double> phases_real(protons.size() * max_steps, 0.0);
        std::vector<double> phases_imag(protons.size() * max_steps, 0.0);

        for (size_t i = 0; i < protons.size(); ++i) {
            const auto& p = protons[i];
            for (size_t step = 0; step < p.TrackPostitions_.size(); ++step) {
                size_t idx_pos = i * max_steps * 3 + step * 3;
                positions_data[idx_pos] = p.TrackPostitions_[step][0];
                positions_data[idx_pos + 1] = p.TrackPostitions_[step][1];
                positions_data[idx_pos + 2] = p.TrackPostitions_[step][2];

                size_t idx_phase = i * max_steps + step;
                phases_real[idx_phase] = p.TrackPhases_[step].real();
                phases_imag[idx_phase] = p.TrackPhases_[step].imag();
            }
        }

        var_positions.putVar(positions_data.data());
        var_phases_real.putVar(phases_real.data());
        var_phases_imag.putVar(phases_imag.data());
        */
       
        dataFile.close();

    } catch (NcException& e) {
        std::cerr << "NetCDF Fehler: " << e.what() << std::endl;
    }
}

void SaveMapsToNETCDF(const std::string& filename,
                         const std::vector<std::vector<std::vector<double>>>& ChiMap,
                         const std::vector<std::vector<std::vector<double>>>& BzMap,
                         double L) {
   

    const size_t n = ChiMap.size();
    if (n == 0) {
        throw std::runtime_error("Empty maps provided.");
    }

    // Flatten 3D arrays into contiguous 1D buffers in row-major order: [i][j][k]
    std::vector<double> chi_flat;
    std::vector<double> bz_flat;
    chi_flat.reserve(n * n * n);
    bz_flat.reserve(n * n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                chi_flat.push_back(ChiMap[i][j][k]);
                bz_flat.push_back(BzMap[i][j][k]);
            }
        }
    }

    try {
        // Create NetCDF file (overwrite if exists)
        NcFile dataFile(filename, NcFile::replace, NcFile::nc4);

        // Define dimensions
        auto dim_x = dataFile.addDim("x", n);
        auto dim_y = dataFile.addDim("y", n);
        auto dim_z = dataFile.addDim("z", n);

        std::vector<NcDim> dims = { dim_x, dim_y, dim_z };

        // Define variables with compression
        NcVar chiVar = dataFile.addVar("ChiMap", ncDouble, dims);
        NcVar bzVar = dataFile.addVar("Bz", ncDouble, dims);

        // Enable chunking and compression (optional but recommended for large 3D data)
        chiVar.setCompression(true, true, 4); // shuffle, deflate, level=4
        bzVar.setCompression(true, true, 4);


        // Write data: NetCDF expects the data in the same order as dimensions (x,y,z)
        // Our flattening was [i][j][k] matching (x,y,z)
        std::vector<size_t> start = {0, 0, 0};
        std::vector<size_t> count = {n, n, n};

        chiVar.putVar(start, count, chi_flat.data());
        bzVar.putVar(start, count, bz_flat.data());


        dataFile.close();

    } catch (const NcException& e) {
        throw std::runtime_error(std::string("NetCDF error: ") + e.what());
    }
}
