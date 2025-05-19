#include <omp.h>
#include <random>

#include "src.hpp"
#include "helper.hpp"

#define GAMMA 267522187


Proton::Proton(const std::vector<int>& position): position_(position), phase_(1.0, 0.0) {};

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

Voxel::Voxel(int n, double L, int SIZE_arti, double DeltaChi, int N, int numberofprotons, std::vector<double> B0, double D, double dt): 
        n_(n), L_(L), SIZE_arti_(SIZE_arti), DeltaChi_(DeltaChi), N_(N),numberofprotons_(numberofprotons), B0_(B0), D_(D), dt_(dt) {
            
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
        std::random_device rd;  // Seed generator
        std::mt19937 gen(rd());
        std::vector<Artifact> artifacts;
        std::uniform_int_distribution<int> dist(0, n_);
        std::vector<int> pos;


        std::cout << "Create Susceptibility Artifacts..." << std::endl;
        for (int i = 0; i < N; i++) {
            pos = {dist(gen), dist(gen), dist(gen)};
            artifacts.push_back(Artifact(pos, 1, SIZE_arti_, DeltaChi_, n_));
        }

        Occupied_positions_ = GetOccupiedPositions(artifacts);

        // Set the susceptibility map
        for (auto& artifact : artifacts) {
            for (auto& pos : artifact.positions_) {
                int x = pos[0] + n_/2;
                int y = pos[1] + n_/2;
                int z = pos[2] + n_/2;
                ChiMap_[x][y][z] += artifact.suscept_;
            }
        }

        std::cout << "Convolve Dipole Kernel with X-map..." << std::endl;

        auto Chimapk_ = applyFFT3D(ChiMap_, n_);
        auto Dk_ = applyFFT3D(dz_, n_);

        //print3DVector(convertComplexToDouble(Dk_));
        //print3DVector(convertComplexToDouble(Dk_));

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

        Protons_init_ = Protons_;

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

std::complex<double> Voxel::SimulateDiffusionSteps(int NrOfSteps, double t) {

        std::random_device rd;
        std::mt19937 gen(rd());

        double dtM_ = dt_ / static_cast<double>(NrOfSteps);

        std::normal_distribution<> dis(0,std::sqrt(2 * D_ * dtM_));

      
        std::vector<std::vector<std::vector<double>>> Paths(numberofprotons_, 
                                                            std::vector<std::vector<double>>(3, std::vector<double>(NrOfSteps, 0.0)));


        #pragma omp parallel for
        for (size_t i = 0; i < numberofprotons_; i++) {
            std::vector<int> position = Protons_[i].position_;
            std::vector<int> candidatePos(3);
            //(std::cout << "Proton: " << i << std::endl;
            
            for (int step = 0; step < NrOfSteps; ++step) {
                bool collision;
                
                do {
                    double dx = dis(gen) * n_ / L_;
                    double dy = dis(gen) * n_ / L_;
                    double dz = dis(gen) * n_ / L_;
        
                    int newX = position[0] + std::round(dx);
                    int newY = position[1] + std::round(dy);
                    int newZ = position[2] + std::round(dz);
        
                    // Periodic boundary conditions
                    newX = ((newX + n_/2) % n_ + n_) % n_ - n_/2;
                    newY = ((newY + n_/2) % n_ + n_) % n_ - n_/2;
                    newZ = ((newZ + n_/2) % n_ + n_) % n_ - n_/2;
        
                    candidatePos[0] = newX;
                    candidatePos[1] = newY;
                    candidatePos[2] = newZ;
                   
                    // Prüfen, ob Position besetzt
                    collision = isElement(Occupied_positions_, candidatePos);
        
                    // Wenn collision == true, wird while nochmal ausgeführt und neuer Schritt probiert
                } while (collision);
        
                // freie Position gefunden -> setzen
                position[0] = candidatePos[0];
                position[1] = candidatePos[1];
                position[2] = candidatePos[2];
        
                Paths[i][0][step] = position[0];
                Paths[i][1][step] = position[1];
                Paths[i][2][step] = position[2];
            }
        
            // Partikelposition aktualisieren
            Protons_[i].position_[0] = position[0]; 
            Protons_[i].position_[1] = position[1];
            Protons_[i].position_[2] = position[2];
            Protons_[i].TrackPostitions_.push_back(Protons_[i].position_);
            
        }   

        
        int x,y,z;
        std::complex<double> totalPhase;
        double omega;
        totalPhase = 0; 
        std::cout << "Calculate Phase..." << std::endl;  

        for(int i = 0; i < numberofprotons_; i++) { 
              // Startwert 1 + 0i
            for (int j = 0; j < NrOfSteps; j++) {
                
                x = static_cast<int>(Paths[i][0][j] + n_ / 2);
                y = static_cast<int>(Paths[i][1][j] + n_ / 2);
                z = static_cast<int>(Paths[i][2][j] + n_ / 2);
                
                omega = GAMMA * (Bz_[x][y][z] + B0_val_);
                
                Protons_[i].phase_ *= std::exp(std::complex<double>(0,- omega *  dtM_));
                
            }
            //Protons_[i].TrackPhases.push_back(Protons_[i].phase_);
           
            totalPhase += Protons_[i].phase_;

            
        }

        std::vector<int> position = Protons_[20].position_;
        
        std::cout << "Position:  " << position[0] << " " << position[1] << " " << position[2] << std::endl;
        totalPhase = totalPhase / static_cast<double>(numberofprotons_);
     
        return totalPhase;

    }

std::complex<double>  Voxel::ComputeSignalStatic(double t) { 
    static int x,y,z;
        std::complex<double> phase, totalPhase;
        double omega;
        totalPhase = 0; 
     
        for(int i = 0; i < numberofprotons_; i++) { 
            x = static_cast<int>(ALL_positions_[i][0] + n_ / 2);
            y = static_cast<int>(ALL_positions_[i][1] + n_ / 2);
            z = static_cast<int>(ALL_positions_[i][2] + n_ / 2);

            omega = GAMMA * (Bz_[x][y][z] + B0_val_); 
         
            phase = std::exp(std::complex<double>(0, -omega * t));
            totalPhase += phase;

        }
        
        return totalPhase / static_cast<double>(numberofprotons_);;

}

std::vector<std::vector<int>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, const std::set<std::vector<int>>& Occupied_positions) {
    std::set<std::vector<int>> final_positions;
    std::vector<std::set<std::vector<int>>> thread_sets(omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(std::random_device{}() + tid);
        std::uniform_real_distribution<double> dist_x(-xlen / 2 + 1, xlen / 2 - 1);
        std::uniform_real_distribution<double> dist_y(-ylen / 2 + 1, ylen / 2 - 1);
        std::uniform_real_distribution<double> dist_z(-zlen / 2 + 1, zlen / 2 - 1);

        size_t target = num_positions / omp_get_num_threads();

        while (thread_sets[tid].size() < target) {
            std::vector<int> pos = {
                static_cast<int>(std::round(dist_x(gen))),
                static_cast<int>(std::round(dist_y(gen))),
                static_cast<int>(std::round(dist_z(gen)))
            };

            if (!Occupied_positions.count(pos) && !thread_sets[tid].count(pos)) {
                thread_sets[tid].insert(pos);
            }
        }
    }

    // Merge all thread-local sets
    for (const auto& s : thread_sets) {
        final_positions.insert(s.begin(), s.end());
    }

    // Fallback: Wenn zu wenige eindeutige Positionen erzeugt wurden
    while (final_positions.size() < static_cast<size_t>(num_positions)) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist_x(-xlen / 2 + 1, xlen / 2 - 1);
        std::uniform_real_distribution<double> dist_y(-ylen / 2 + 1, ylen / 2 - 1);
        std::uniform_real_distribution<double> dist_z(-zlen / 2 + 1, zlen / 2 - 1);

        std::vector<int> pos = {
            static_cast<int>(std::round(dist_x(gen))),
            static_cast<int>(std::round(dist_y(gen))),
            static_cast<int>(std::round(dist_z(gen)))
        };

        if (!Occupied_positions.count(pos)) {
            final_positions.insert(pos);
        }
    }

    return std::vector<std::vector<int>>(final_positions.begin(), final_positions.end());
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

