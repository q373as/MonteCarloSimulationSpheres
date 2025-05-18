#include <iostream>
#include <string>
#include <fstream>

#include "helper.hpp"


std::vector<std::vector<std::vector<double>>> applyIFFT3D(const std::vector<std::vector<std::vector<std::complex<double>>>>& kSpace, int n, double B0_val) {
    std::vector<std::vector<std::vector<double>>> rSpace(n, std::vector<std::vector<double>>(n, std::vector<double>(n)));

    // Erstelle den FFTW-Plan für die inverse 3D-Transformation
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);


    // Übertrage die Daten aus kSpace in das fftw_complex Array
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                in[i * n * n + j * n + k][0] = kSpace[i][j][k].real();  // Realteil
                in[i * n * n + j * n + k][1] = kSpace[i][j][k].imag();  // Imaginärteil
            }

     

    // Erstelle den FFTW-Plan für die inverse Transformation
    fftw_plan plan = fftw_plan_dft_3d(n, n, n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Wende den FFT-Plan an
    fftw_execute(plan);

    // Übertrage die transformierten Daten in das ChiMap_-Array
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                double shift = (out[i * n * n + j * n + k][0]);
                rSpace[i][j][k] = B0_val * shift / (n * n * n);  // Normalisierung
            }

    // Plan und Speicher freigeben
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return rSpace;  // Rückgabe der zurücktransformierten ChiMap_
}

std::vector<std::vector<std::vector<std::complex<double>>>> applyFFT3D(const std::vector<std::vector<std::vector<double>>>& rSpace, int n) {
    std::vector<std::vector<std::vector<std::complex<double>>>> kSpace(n, std::vector<std::vector<std::complex<double>>>(n, std::vector<std::complex<double>>(n)));

    // Erstelle den FFTW-Plan für 3D-Daten
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n * n * n);

    for (int i = 0; i < n * n * n; ++i) {
        in[i][0] = 0.0;  // Real part
        in[i][1] = 0.0;  // Imaginary part
    }

    // Übertrage die Daten aus ChiMap_ in das fftw_complex Array
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                in[i * n * n + j * n + k][0] = rSpace[i][j][k];  // Realteil
                in[i * n * n + j * n + k][1] = 0.0;  // Imaginärteil

            }

    
    // Erstelle den FFTW-Plan
    fftw_plan plan = fftw_plan_dft_3d(n, n, n, in, out, FFTW_FORWARD, FFTW_MEASURE);

    // Wende den FFT-Plan an
    fftw_execute(plan);

    // Übertrage die transformierten Daten in das kSpace-Array
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                kSpace[i][j][k] = std::complex<double>(out[i * n * n + j * n + k][0], out[i * n * n + j * n + k][1]);
            }

    // Plan und Speicher freigeben
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return kSpace;  // Rückgabe des k-space
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

bool isElement(std::vector<std::vector<int>>& outer, std::vector<int>& inner) {
    for (const auto& elem : outer) {
        // Compare each vector
        if (elem == inner) {
            return true;
        }
    }
    return false;
}

void SaveSignalDecay(const std::vector<double>& times, const std::vector<double>& magnitudes, const std::vector<double>& signal, const std::vector<double>& star, std::string filename)
        {


        std::ofstream outFile(filename);


        if (!outFile.is_open()) {
        std::cerr << "Error opening signal_decay file for writing!\n";
        return;
        }

        outFile << "Time\tStaticDephasing\tDiffusion\tAnalytic\n";
        for (size_t i = 0; i < times.size(); ++i) {
                outFile << times[i] << "\t" 
                << magnitudes[i] << "\t" 
                << signal[i] << "\t" 
                << star[i] << "\n";
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
    