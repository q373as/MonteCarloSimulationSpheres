#include <omp.h>
#include <random>
#include <fstream>
#include "src.hpp"
#include "helper.hpp"

#define GAMMA 267522187


Proton::Proton(const std::vector<double>& position): initial_position_(position), phase_(1.0, 0.0) {
    position_ = initial_position_;
};

Artifact::Artifact(std::vector<int> positionmain, int pm, int size, int n) {
        positionmain_ = positionmain;
        size_ = size;
       
            int x0 = positionmain_[0];
            int y0 = positionmain_[1];
            int z0 = positionmain_[2];
            int radius = size;

            for (int x = -size; x < size; x++) {
                for (int y = -size; y < size; y++) {
                    for (int z = -size; z < size; z++) {
                        int r = x * x + y * y + z * z;
                        if (r <= radius * radius) {
                            // Apply periodic boundary conditions
                            int wrapped_x = ((x + x0 + n) % n) - n/2;
                            int wrapped_y = ((y + y0 + n) % n) - n/2;
                            int wrapped_z = ((z + z0 + n) % n) - n/2;

                            positions_.push_back({wrapped_x, wrapped_y, wrapped_z});
                        }
                    }
                }
            }
           
        }
       
Voxel::Voxel(int n, double L, double Xtot, double eta, int numberofprotons,
             std::vector<double> B0, double D, double dt, double mu, double sigma, double ratio)
    : n_(n), L_(L), Xtot_(Xtot), eta_(eta),
      numberofprotons_(numberofprotons), B0_(std::move(B0)),
      D_(D), dt_(dt), mu_(mu), sigma_(sigma), ratio_(ratio) {

    std::cout << "\nStart Monte Carlo SIMULATOR\n\nInitialize Voxel for Simulation..." << std::endl;

    // Grundlegende Validierungen
    if (n_ <= 0) {
        throw std::invalid_argument("n muss > 0 sein");
    }
    if (L_ <= 0.0) {
        throw std::invalid_argument("L muss > 0 sein");
    }
    if (numberofprotons_ <= 0) {
        throw std::invalid_argument("Anzahl der Protonen muss > 0 sein");
    }
    if (B0_.size() < 3) {
        throw std::invalid_argument("B0 muss mindestens 3 Komponenten enthalten");
    }

    // Initialisiere 3D-Felder mit Null
    Dk_ = std::vector<std::vector<std::vector<std::complex<double>>>>(n_, std::vector<std::vector<std::complex<double>>>(n_, std::vector<std::complex<double>>(n_, std::complex<double>(0.0, 0.0))));

    ChiMap_ = std::vector<std::vector<std::vector<double>>>(n_, std::vector<std::vector<double>>(n_, std::vector<double>(n_, 0.0)));

    Bz_ = std::vector<std::vector<std::vector<double>>>(n_, std::vector<std::vector<double>>(n_, std::vector<double>(n_, 0.0)));

    std::cout << "Calculate Dz-Map..." << std::endl;
    

    std::cout << "Set Random Seed Generator..." << std::endl;
    std::mt19937 gen(12345); // feste Seed für Reproduzierbarkeit; ggf. extern machen

    std::uniform_int_distribution<int> dist(0, n_ - 1); // vermeiden von out-of-bounds
    std::lognormal_distribution<double> lognorm_dist(mu_, sigma_);

    std::cout << "Create Susceptibility Artifacts..." << std::endl;

    std::vector<int> pos;
    double Radius = 0.0;
    double Volf = 0.0;
    double XtotCheck = 0.0;

    // Sicherstellen, dass Zähler initialisiert ist (falls nicht bereits als Member)
    N_ = 0;

    while (Volf < eta_) {
        // Radius in physikalischer Einheit (z.B. µm), abhängig von Interpretation
        double radius_sample = lognorm_dist(gen);
        int radius_discrete = static_cast<int>(std::lround(radius_sample * 1e-6 * n_ / L_));

        pos = { dist(gen), dist(gen), dist(gen) };
        Artifact artifact(pos, 1, radius_discrete, n_);

        // Volumenanteil dieses Artefakts: Anzahl Positionen * Einzelvolumen
        artifact.etai_ = static_cast<double>(artifact.positions_.size()) * Vm_;

        Volf += artifact.etai_;

        std::cout << N_ << " Artifact with Radius (µm): " << radius_sample
                  << " | discrete radius: " << radius_discrete << std::endl;

        artifacts.push_back(artifact);
        N_ += 1;
    }

    std::cout << "Real Volfrac: " << Volf << std::endl;
    std::cout << "Add Susceptibility Distribution..." << std::endl;

    Occupied_positions_ = GetOccupiedPositions(artifacts);

    // Vermeide Division durch Null
    if (Occupied_positions_.empty()) {
        throw std::runtime_error("Keine besetzten Positionen gefunden, Suszeptibilitätsverteilung fehlgeschlagen.");
    }

   // Anteil positiver Quellen (zwischen 0 und 1, aber != 0.5)
    double etaP = ratio * eta_;
    double etaN = (1.0 - ratio) * eta_;

    // DeltaChi so wählen, dass Bulk = Xtot
    double denom = eta_ * (2.0 * ratio - 1.0);

    if (std::abs(denom) < 1e-12) {
        throw std::runtime_error("Ungültiges ratio=0.5 für gegebenes Xtot!");
    }


    // Gesamtzahl aller Positionen bestimmen
    size_t total_positions = Occupied_positions_.size();

        
    double DeltaChi = Xtot_ / denom;
    double chiP =  DeltaChi;
    double chiN = -DeltaChi;

    // Grenze für positive Quellen
    size_t nArtifacts = artifacts.size();
    size_t nPosArtifacts = static_cast<size_t>(ratio * nArtifacts);

    size_t count = 0;
    for (const auto& artifact : artifacts) {
        // Entscheide, ob das ganze Artefakt positiv oder negativ ist
        double chi = (count < nPosArtifacts) ? chiP : chiN;

        for (const auto& p : artifact.positions_) {
            int x = p[0] + n_ / 2;
            int y = p[1] + n_ / 2;
            int z = p[2] + n_ / 2;
            if (x < 0 || x >= n_ || y < 0 || y >= n_ || z < 0 || z >= n_) continue;

            // Nur setzen, wenn der Voxel noch leer ist
            if (ChiMap_[x][y][z] == 0.0) {
                ChiMap_[x][y][z] = chi;
            }
        }
        count++;
    }



    std::cout << "Susceptibility per Artifact: " << Xtot_ / (eta_) << std::endl;
    std::cout << "Convolve Dipole Kernel with X-map..." << std::endl;

    auto Chimapk_ = applyFFT3D(ChiMap_, n_);

    int mid = n_ / 2;
    double invNdx = 1.0 / (static_cast<double>(n_));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            for (int k = 0; k < n_; ++k) {
                int ii = (i <= n_/2) ? i : i - n_;
                int jj = (j <= n_/2) ? j : j - n_;
                int kk = (k <= n_/2) ? k : k - n_;

                double kx = static_cast<double>(ii) * invNdx;
                double ky = static_cast<double>(jj) * invNdx;
                double kz = static_cast<double>(kk) * invNdx;

                double k2 = kx*kx + ky*ky + kz*kz;
                double val = 0.0;

                if (k2 > 0.0) {
                    double kdotn = (kx * (B0_[0] / B0_val_)) +
                                (ky * (B0_[1] / B0_val_)) +
                                (kz * (B0_[2] / B0_val_));
                    val = 1.0/3.0 - (kdotn * kdotn) / k2;
                }

                Dk_[i][j][k] = std::complex<double>(val, 0.0);
            }
        }
    }

    auto Dk_plot = Dk_;
    shift3DArray(Dk_plot, n_);    

    auto multip_ = Dk_; 

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            for (int k = 0; k < n_; ++k) {
                multip_[i][j][k] *= Chimapk_[i][j][k];
            }
        }
    }

    Bz_ = applyIFFT3D(multip_, n_, B0_val_);
    
    std::cout << "Initialize Protons..." << std::endl;

    ALL_positions_ = GetALL_positions(n_, n_, n_, numberofprotons_, Occupied_positions_);

    if (static_cast<int>(ALL_positions_.size()) < numberofprotons_) {
        throw std::runtime_error("Nicht genügend Startpositionen für alle Protonen generiert.");
    }

    Protons_.reserve(numberofprotons_);
    
    for (int i = 0; i < numberofprotons_; ++i) {
        Protons_.emplace_back(Proton(ALL_positions_[i]));
    }


    std::cout << "\nVoxel Initialization Done...\n" << std::endl;
}

std::tuple<
    std::vector<std::vector<std::vector<double>>>, 
    std::vector<std::vector<std::vector<double>>>, 
    std::vector<std::vector<std::vector<double>>>>
Voxel::ComputeSpatialCorrelations3D() 
{
    std::cout << "Compute full 3D spatial correlation functions..." << std::endl;

    // Extra Maps für positive und negative Quellen
    auto ChiMapPos = ChiMap_;
    auto ChiMapNeg = ChiMap_;
    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            for (int k = 0; k < n_; k++) {
                if (ChiMap_[i][j][k] > 0.0) {
                    ChiMapNeg[i][j][k] = 0.0;
                } else {
                    ChiMapPos[i][j][k] = 0.0;
                }
            }
        }
    }

    // FFT von Chi+ und Chi-
    auto Fpos = applyFFT3D(ChiMapPos, n_);
    auto Fneg = applyFFT3D(ChiMapNeg, n_);

    // Container für Correlation Spectra
    auto Cpp_k = Fpos;
    auto Cnn_k = Fneg;
    auto Cpn_k = Fpos;

    for (int i = 0; i < n_; i++) {
        for (int j = 0; j < n_; j++) {
            for (int k = 0; k < n_; k++) {
                std::complex<double> a = Fpos[i][j][k];
                std::complex<double> b = Fneg[i][j][k];

                Cpp_k[i][j][k] = a * std::conj(a); // |Fpos|^2
                Cnn_k[i][j][k] = b * std::conj(b); // |Fneg|^2
                Cpn_k[i][j][k] = a * std::conj(b); // Kreuz
            }
        }
    }

    // Rücktransformation ins Real-Space
    auto Cxx_pp = applyIFFT3D(Cpp_k, n_, 1.0);
    auto Cxx_nn = applyIFFT3D(Cnn_k, n_, 1.0);
    auto Cxx_pn = applyIFFT3D(Cpn_k, n_, 1.0);

    std::cout << "3D correlation maps computed." << std::endl;

    return {Cxx_pp, Cxx_nn, Cxx_pn};
}


double Voxel::interpolateBz(double x, double y, double z) {
    int x0 = static_cast<int>(std::floor(x)) % n_;
    int x1 = (x0 + 1) % n_;

    int y0 = static_cast<int>(std::floor(y)) % n_;
    int y1 = (y0 + 1) % n_;

    int z0 = static_cast<int>(std::floor(z)) % n_;
    int z1 = (z0 + 1) % n_;


    double xd = x - x0, yd = y - y0, zd = z - z0;

    double c00 = Bz_[x0][y0][z0] * (1 - xd) + Bz_[x1][y0][z0] * xd;
    double c01 = Bz_[x0][y0][z1] * (1 - xd) + Bz_[x1][y0][z1] * xd;
    double c10 = Bz_[x0][y1][z0] * (1 - xd) + Bz_[x1][y1][z0] * xd;
    double c11 = Bz_[x0][y1][z1] * (1 - xd) + Bz_[x1][y1][z1] * xd;

    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}

SignalResults Voxel::SimulateDiffusionSteps(int NrOfSteps) {    
        double steps = static_cast<double>(NrOfSteps);
        double dtM_ = dt_ / steps;
        std::vector<std::vector<std::vector<double>>> Paths(numberofprotons_, 
                                                            std::vector<std::vector<double>>(3, std::vector<double>(NrOfSteps, 0.0)));

        #pragma omp parallel for
        for (size_t i = 0; i < numberofprotons_; i++) {
            std::vector<double> position = Protons_[i].position_;
            std::vector<double> candidatePos(3);
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::normal_distribution<> dis(0,std::sqrt(2 * D_ * dtM_) * n_ / L_);

            for (int step = 0; step < NrOfSteps; ++step) {
                bool collision;
                
                do {
                    double dx = dis(gen);
                    double dy = dis(gen);
                    double dz = dis(gen);
                    
                    
                    candidatePos[0] = position[0] + dx;
                    candidatePos[1] = position[1] + dy;
                    candidatePos[2] = position[2] + dz;
        
                    
                    auto wrap = [this](double val) -> double {
                        double halfN = n_ / 2.0;
                        double res = std::fmod(val + halfN, n_);
                        if (res < 0) res += n_;
                        return res - halfN;
                    };

                    candidatePos[0] = wrap(candidatePos[0]);
                    candidatePos[1] = wrap(candidatePos[1]);
                    candidatePos[2] = wrap(candidatePos[2]);
        
                    collision = isElement(Occupied_positions_, candidatePos);
        
                } while (collision);
        
              
                position = candidatePos;
        
                Paths[i][0][step] = position[0];
                Paths[i][1][step] = position[1];
                Paths[i][2][step] = position[2];
            }
        
            
            Protons_[i].position_ = position;
            Protons_[i].TrackPostitions_.push_back(position);
        }

        
        int x,y,z;
        std::complex<double> totalPhase;
        std::vector<double> all_phases;
        all_phases.reserve(numberofprotons_);
        double omega;
        totalPhase = 0; 
        
        for(int i = 0; i < numberofprotons_; i++) { 
            for (int j = 0; j < NrOfSteps; j++) {

                omega = GAMMA * (interpolateBz(Paths[i][0][j] + n_/2.0, Paths[i][1][j] + n_/2.0, Paths[i][2][j] + n_/2.0)) / ( 2* M_PI);

                Protons_[i].phase_ *= std::exp(std::complex<double>(0,- omega *  dtM_));
                all_phases.push_back(std::arg(Protons_[i].phase_));
            }
            
            totalPhase += Protons_[i].phase_;
            Protons_[i].TrackPhases_.push_back(Protons_[i].phase_);
        }
        

        totalPhase = totalPhase / static_cast<double>(numberofprotons_);
     
        double m1 = 0.0;
        for (double phi : all_phases) m1 += phi;
        m1 /= all_phases.size();

        double m2 = 0, m3 = 0, m4 = 0;
        for (double phi : all_phases) {
            double dphi = phi - m1;
            m2 += dphi * dphi;
            m3 += dphi * dphi * dphi;
            m4 += dphi * dphi * dphi * dphi;
        }
        m2 /= all_phases.size();
        m3 /= all_phases.size();
        m4 /= all_phases.size();

   
        double kappa1 = m1;              // Mittelwert
        double kappa2 = m2;                // Varianz
        double kappa3 = m3;                // Schiefe 
        double kappa4 = m4 - 3 * m2 * m2;  // Kurtosis 


        std::complex<double> signal_kappa_2 = std::exp(std::complex<double>(0, kappa1) - 0.5 * kappa2);

        std::complex<double> signal_kappa_4 = std::exp(std::complex<double>(0, kappa1)
            - 0.5 * kappa2
            + std::complex<double>(0, 1.0/6.0) * kappa3
            - 1.0/24.0 * kappa4
        );

        std::vector<double> moments = {m1, m2, m3, m4};
        std::vector<double> cumulants = {kappa1, kappa2, kappa3, kappa4};
        
        SignalResults results {
                signal_kappa_2,
                signal_kappa_4,
                moments,
                cumulants,
                totalPhase
            };

     
        return results;
    }

std::complex<double> Voxel::SimulateSpinEchoSignal(int diffusionSteps, double dt, double TE) {
    double current_time = 0.0;
    int t = 0;
    int TE_half_step = static_cast<int>(std::round(TE / 2.0 / dt));

    // Reset Protonen
    for (auto& p : Protons_) {
        p.phase_ = std::complex<double>(1.0, 0.0);
        p.position_ = p.initial_position_;
        p.TrackPhases_.clear();
        p.TrackPostitions_.clear();
    }

    bool pulse_applied = false;

    while (current_time <= TE) {
        this->SimulateDiffusionSteps(diffusionSteps);

        // Prüfe, ob wir über die Hälfte der Echozeit hinaus sind und der Puls noch nicht angewendet wurde
        if (!pulse_applied && current_time >= TE / 2.0) {
            for (auto& p : Protons_) {
                p.phase_ = std::conj(p.phase_);
            }
            pulse_applied = true;
            std::cout << "π-Puls applied at t = " << current_time << std::endl;
        }

        // Signal bei Echozeit zurückgeben
        if (std::abs(current_time - TE) < 0.5 * dt || current_time > TE) {
            std::complex<double> avg_signal = 0.0;
            for (auto& p : Protons_) {
                avg_signal += p.phase_;
            }
            avg_signal /= static_cast<double>(numberofprotons_);
            return avg_signal;
        }

        ++t;
        current_time = t * dt;
    }


    throw std::runtime_error("TE exceeds simulation time range.");
}

std::vector<double> Voxel::ComputeTemporalACF(int tsteps) {
    std::vector<double> acf(tsteps, 0.0);
    std::vector<int> count(tsteps, 0);

    double omegaMean = 0.0;
    double totalCount = 0;
    for (const auto& p : Protons_) {
        for (const auto& pos : p.TrackPostitions_) {

            omegaMean += interpolateBz(pos[0] + n_/2, pos[1] + n_/2, pos[2] + n_/2);
            totalCount++;
        }
    }

    omegaMean /= totalCount;

    #pragma omp parallel for
    for (int lag = 0; lag < tsteps; ++lag) {
        double sum = 0.0;
        int validCount = 0;

        double localSum = 0.0;
        int localCount = 0;

        for (const auto& p : Protons_) {
            int T_p = static_cast<int>(p.TrackPostitions_.size());
            for (int t = 0; t + lag < T_p; ++t) {
                double x1 = p.TrackPostitions_[t][0] + n_ / 2;
                double y1 = p.TrackPostitions_[t][1] + n_ / 2;
                double z1 = p.TrackPostitions_[t][2] + n_ / 2;
                double x2 = p.TrackPostitions_[t + lag][0] + n_ / 2;
                double y2 = p.TrackPostitions_[t + lag][1] + n_ / 2;
                double z2 = p.TrackPostitions_[t + lag][2] + n_ / 2;

                double omega1 = interpolateBz(x1, y1, z1) - omegaMean;
                double omega2 = interpolateBz(x2, y2, z2) - omegaMean;

                localSum +=  omega1 * omega2;
                localCount++;
            }
        }

        localSum = localSum * GAMMA * GAMMA;
        acf[lag] = (localCount > 0) ? (localSum / localCount) : 0.0;
    }

    return acf;
}

std::complex<double>  Voxel::ComputeSignalStatic(double t) { 
    static int x,y,z;
        std::complex<double> phase, totalPhase;
        double omega;
        totalPhase = 0; 
     
        for(int i = 0; i < numberofprotons_; i++) { 
            x = static_cast<int>(std::round(ALL_positions_[i][0] + n_/2.0));
            y = static_cast<int>(std::round(ALL_positions_[i][1] + n_/2.0));
            z = static_cast<int>(std::round(ALL_positions_[i][2] + n_/2.0));

            omega =   GAMMA * (Bz_[x][y][z]) / (2 * M_PI); 
         
            phase = std::exp(std::complex<double>(0, -omega * t));
            totalPhase += phase;

        }
        
        return totalPhase / static_cast<double>(numberofprotons_);
    }

std::vector<std::vector<double>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, const std::set<std::vector<int>>& Occupied_positions) {
    std::set<std::vector<double>> final_positions;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_x(-xlen / 2.0 + 1, xlen / 2.0 - 1);
    std::uniform_real_distribution<double> dist_y(-ylen / 2.0 + 1, ylen / 2.0 - 1);
    std::uniform_real_distribution<double> dist_z(-zlen / 2.0 + 1, zlen / 2.0 - 1);

    while (final_positions.size() < static_cast<size_t>(num_positions)) {
        std::vector<double> pos = {dist_x(gen), dist_y(gen), dist_z(gen)};

        // Prüfen, ob die gerundete Position schon belegt ist oder wir sie schon haben
        if (!isElement(Occupied_positions, pos) && final_positions.count(pos) == 0) {
            final_positions.insert(pos);
        }
    }

    return std::vector<std::vector<double>>(final_positions.begin(), final_positions.end());
}

std::set<std::vector<int>> GetOccupiedPositions(const std::vector<Artifact>& artifacts) {
    std::set<std::vector<int>> all_positions;

    // Iterate through all artifacts and collect positions
    for (const auto& artifact : artifacts) {
        for (const auto& pos : artifact.positions_) {
            all_positions.insert(pos);  // statt push_back
        }
    }

    return all_positions;
}

