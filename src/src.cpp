#include <omp.h>
#include <random>

#include "src.hpp"
#include "helper.hpp"

#define GAMMA 267522187


Proton::Proton(const std::vector<double>& position): position_(position), phase_(1.0, 0.0) {};

Artifact::Artifact(std::vector<int> positionmain, int pm, int size, double suscept, int n) {
    positionmain_ = positionmain;
    
            size_ = size;
            int x0 = positionmain_[0];
            int y0 = positionmain_[1];
            int z0 = positionmain_[2];
            int radius = size;
            suscept_ = suscept;

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

            totsuscept_ = pm * suscept_ * positions_.size();
        }

Voxel::Voxel(int n, double L, double Xtot, double eta, int numberofprotons, std::vector<double> B0, double D, double dt, double mu, double sigma): 
        n_(n), L_(L), Xtot_(Xtot), eta_(eta), numberofprotons_(numberofprotons), B0_(B0), D_(D), dt_(dt),mu_(mu), sigma_(sigma)  {
            
        std::cout << "\nStart Monte Carlo SIMULATOR\n\nInitialize Voxel for Simulation..."  << std::endl;
            
        dz_ = std::vector<std::vector<std::vector<double>>>(n, 
            std::vector<std::vector<double>>(n, 
            std::vector<double>(n, 0.0)));

        ChiMap_ = std::vector<std::vector<std::vector<double>>>(n,std::vector<std::vector<double>>(n, 
                            std::vector<double>(n, 0.0)));

        
        Bz_ = std::vector<std::vector<std::vector<double>>>(n, 
            std::vector<std::vector<double>>(n, 
            std::vector<double>(n, 0.0)));

        
        std::cout << "Calculate Dz-Map..." << std::endl;

        CalculateDzMap();

        std::cout << "Set Random Seed Generator..." << std::endl;
        std::random_device rd; 
        std::mt19937 gen(rd());
        
        std::uniform_int_distribution<int> dist(0, n_);
        std::lognormal_distribution<double> lognorm_dist(mu_, sigma_);
    
        std::cout << "Create Susceptibility Artifacts..." << std::endl;
        
        std::vector<int> pos;
        double Radius, Volf = 0.0;
        
        while (Volf < eta_) {

            double Radius = lognorm_dist(gen) * 1e-6 * n_ / L_;
            int radius_discrete = std::round(Radius);
    
            pos = {dist(gen), dist(gen), dist(gen)};
            Artifact artifact(pos, 1, radius_discrete, DeltaChi_, n_);

            Volf += static_cast<double>(artifact.positions_.size()) * Vm_;
            
            artifacts.push_back(artifact);
            N_ += 1;

        }
        

        Occupied_positions_ = GetOccupiedPositions(artifacts);

        double sus = Xtot_ / Occupied_positions_.size();

        for (auto& artifact : artifacts) {
            
            for (auto& pos : artifact.positions_) {
                int x = pos[0] + n_/2;
                int y = pos[1] + n_/2;
                int z = pos[2] + n_/2;
                ChiMap_[x][y][z] += sus;
            }
        }

        std::cout << "Convolve Dipole Kernel with X-map..." << std::endl;

        auto Chimapk_ = applyFFT3D(ChiMap_, n_);
        auto Dk_ = applyFFT3D(dz_, n_);

        auto multip_ = Dk_;

        #pragma omp parallel for collapse(3)
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                for (int k = 0; k < n_; ++k) {
                    multip_[i][j][k] *=  Chimapk_[i][j][k];
                    }
                }
            }

        Bz_ = applyIFFT3D(multip_,n_, B0_[2]);
        
        std::cout << "Initialize Protons..." << std::endl;

        ALL_positions_ = GetALL_positions(n_,n_,n_,numberofprotons_, Occupied_positions_);

        for (int i = 0; i < numberofprotons_; i++) {
            Protons_.push_back(Proton(ALL_positions_[i]));
        }

        shift3DArray(Bz_, n_);
      
        std::cout << "\nVoxel Initialization Done...\n" << std::endl;

            

}

void Voxel::CalculateDzMap() {
    #pragma omp parallel for collapse(3)
        for (int x = -n_ / 2; x < n_/2; x++) {
            for (int y = -n_ / 2; y < n_/2; y++) {
                for (int z = -n_ / 2; z < n_ / 2; z++) {
                   
                    std::vector<double> position = {x / nb_, y / nb_, z / nb_};

                    double r = sqrt(position[0] * position[0] + position[1] * position[1] + position[2] * position[2]);
                    double B0b = sqrt(B0_[0] * B0_[0] + B0_[1] * B0_[1] + B0_[2] * B0_[2]);
                    
                    double costheta = (B0_[0] * position[0] + B0_[1] * position[1] + B0_[2] * position[2]) / (B0b * r);
                    double dz = (1.0f / (4.0f * M_PI)) * ((3 * costheta * costheta - 1) / (r * r * r));
                    
                    if(r == 0) dz_[x + n_ / 2][y + n_/2][z + n_/2] = 0;
                    else   dz_[x + n_ / 2][y + n_/2][z + n_/2] += dz;     
                }
            }
        }
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

SignalResults Voxel::SimulateDiffusionSteps(int NrOfSteps, double t) {    
        double steps = static_cast<double>(NrOfSteps);
        double dtM_ = dt_ / steps;
        std::vector<std::vector<std::vector<double>>> Paths(numberofprotons_, 
                                                            std::vector<std::vector<double>>(3, std::vector<double>(NrOfSteps, 0.0)));

        #pragma omp parallel for
        for (size_t i = 0; i < numberofprotons_; i++) {
            std::vector<double> position = Protons_[i].position_;
            std::vector<double> candidatePos(3);
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num()); // eigenständiger RNG pro Thread
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
        
                    // Periodische Randbedingungen für double Werte
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
        
                // freie Position gefunden -> setzen (mit double-Werten)
                position = candidatePos;
        
                Paths[i][0][step] = position[0];
                Paths[i][1][step] = position[1];
                Paths[i][2][step] = position[2];
            }
        
            // Partikelposition aktualisieren
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

                omega = GAMMA * (interpolateBz(Paths[i][0][j] + n_/2.0, Paths[i][1][j] + n_/2.0, Paths[i][2][j] + n_/2.0));

                Protons_[i].phase_ *= std::exp(std::complex<double>(0,- omega *  dtM_));
                all_phases.push_back(std::arg(Protons_[i].phase_));
            }
            
            totalPhase += Protons_[i].phase_;

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
        double kappa3 = m3;                // Schiefe * Varianz^1.5
        double kappa4 = m4 - 3 * m2 * m2;  // Kurtosis * Varianz^2


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

                localSum += omega1 * omega2;
                localCount++;
            }
        }

     
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

            omega =   GAMMA * (Bz_[x][y][z]); 
         
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


