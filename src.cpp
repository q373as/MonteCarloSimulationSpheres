#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <fftw3.h>
#include <omp.h>
#include <ctime>
#include <random>
#include <complex>

#include <matplotlib-cpp/matplotlibcpp.h>

#define GAMMA 267522187

 // Set the number of threads you want to use

namespace plt = matplotlibcpp;



void plotBzWithParticles(std::vector<std::vector<std::vector<double>>> Bz, 
                         std::vector<std::vector<int>>& particles, 
                         int n) {
    // Nehmen wir an, wir wollen einen 2D-Schnitt von Bz plotten (z.B. Bz[i][j])
    int slice = 0; // Mittlere Schicht als Beispiel
    std::vector<std::vector<double>> Bz_slice(n, std::vector<double>(n));
   
    // Extrahiere einen 2D-Slice von Bz entlang der z-Achse (Bz[i][j][slice])
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Bz_slice[i][j] = Bz[j][i][slice]; // Slice durch die dritte Dimension
        }
    }

    // Umwandlung von 2D std::vector in flachen Array
    std::vector<float> flat_Bz;
    flat_Bz.reserve(n * n);  // Vorreservieren der Speichergröße für Effizienz
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat_Bz.push_back(static_cast<float>(Bz_slice[i][j]));  // Umwandeln in float
        }
    }
    // Plotten der Bz-Daten als Heatmap
    plt::imshow(flat_Bz.data(), n, n, 1, {{"cmap", "coolwarm"}}); // Setze die Farben auf "YlGn"

    // Extrahiere x- und y-Koordinaten der Partikel
    std::vector<int> particle_x, particle_y;
    for (const auto& particle : particles) {
        int x = static_cast<int>(particle[0] + n/2);  // x-Koordinate
        int y = static_cast<int>(particle[1] + n/2 );  // y-Koordinate
        
        // Überprüfen, ob die Partikel innerhalb des Grid liegen
        if (static_cast<int> (particle[2] + n/2) == slice) {
            particle_x.push_back(static_cast<double>(x));
            particle_y.push_back(static_cast<double>(y));
           
        }
    }


    // Partikel als schwarze Punkte plotten
    double alpha = 0.5;  // 50% transparency
    
    // Scatter plot with alpha as part of the color
    plt::scatter(particle_x, particle_y, 1, {{"color", "black"}});
    // Optional: Hinzufügen eines Colorbars
    //plt::colorbar();  // Optional: füge eine Farblegende hinzu
    plt::title("Bz Distribution with Particle Positions");
    plt::show();
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
                rSpace[i][j][k] = B0_val * out[i * n * n + j * n + k][0] / (n * n * n);  // Normalisierung
            }

    // Plan und Speicher freigeben
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return rSpace;  // Rückgabe der zurücktransformierten ChiMap_
}

void print3DVector(const std::vector<std::vector<std::vector<double>>>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        for (size_t j = 0; j < vec[i].size(); j++) {
            for (size_t k = 0; k < vec[i][j].size(); k++) {
                std::cout << "vec[" << i << "][" << j << "][" << k << "] = " 
                          << vec[i][j][k] << std::endl;
            }
        }
    }
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


std::vector<std::vector<int>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, std::vector<std::vector<int>> Occupied_positions) {
    std::vector<std::vector<int>> positions;
    positions.reserve(num_positions);  // Reserve space for the positions

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_x(-xlen /2 + 1, xlen / 2 - 1);
    std::uniform_real_distribution<double> dist_y(- ylen/ 2 + 1, ylen / 2 - 1);
    std::uniform_real_distribution<double> dist_z(-zlen / 2 + 1, zlen / 2- 1);

    #pragma omp parallel
    {
        std::mt19937 thread_gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<double> thread_dist_x(-xlen /2 + 1, xlen /2 - 1);
        std::uniform_real_distribution<double> thread_dist_y(-ylen / 2 + 1, ylen / 2 - 1);
        std::uniform_real_distribution<double> thread_dist_z(-zlen / 2 + 1, zlen / 2 - 1);

        #pragma omp for
        for (int i = 0; i < num_positions; i++) {
            std::vector<int> new_position;
            bool is_unique = false;

            while (!is_unique) {
                // Generate a new random position
                new_position = {
                    static_cast<int>(std::round(thread_dist_x(thread_gen))),
                    static_cast<int>(std::round(thread_dist_y(thread_gen))),
                    static_cast<int>(std::round(thread_dist_z(thread_gen)))
                };


                // Check if the position is already occupied
                is_unique = !isElement(Occupied_positions, new_position);
            }

            #pragma omp critical
            {   
                positions.push_back(new_position);  // Add the unique position to the results
            }
        }
    }

    return positions;
}

class Artifact {
    public:
        std::vector<int> positionmain_;
        double suscept_;
        int size_;
        std::vector<std::vector<int>> positions_;

        Artifact(std::vector<int> positionmain, int pm, int size, double suscept, int n) {
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
            suscept_ = pm * suscept * positions_.size();
        }
};

std::vector<std::vector<int>> GetOccupiedPositions(std::vector<Artifact>& artifacts) {
    std::vector<std::vector<int>> all_positions;

    // Iterate through all artifacts and collect positions
    for (const auto& artifact : artifacts) {
        for (const auto& pos : artifact.positions_) {
            all_positions.push_back(pos);
        }
    }

    return all_positions;
}

class Voxel {
    public:
        int n_;
        double L_;
        int N_;
        int SIZE_arti_;
        double DeltaChi_;
        int numberofprotons_;
        double B0_val_;
        double dt_;
        double D_;

        std::vector<double> B0_;
        std::vector<std::vector<std::vector<double>>> dz_;
        std::vector<std::vector<std::vector<double>>> ChiMap_;
        std::vector<std::vector<std::vector<double>>> Bz_;
        std::vector<std::vector<int>> ALL_positions_, ALL_positions_init_;
        std::vector<std::vector<int>> Occupied_positions_;

        fftw_complex* dz_fftw_;
        fftw_complex* ChiMap_fftw_;
        fftw_complex* Bz_fftw_;

        Voxel(int n, double L, int SIZE_arti, double DeltaChi, int N, int numberofprotons, std::vector<double> B0, double dt): 
            n_(n), L_(L), SIZE_arti_(SIZE_arti), DeltaChi_(DeltaChi), N_(N),numberofprotons_(numberofprotons), B0_(B0), dt_(dt) {


            D_ = 0; 
            dt_ = 0.0001;
            dz_ = std::vector<std::vector<std::vector<double>>>(n, 
                std::vector<std::vector<double>>(n, 
                std::vector<double>(n, 0.0)));

            ChiMap_ = std::vector<std::vector<std::vector<double>>>(n,std::vector<std::vector<double>>(n, 
                                std::vector<double>(n, 0.0)));

            
            Bz_ = std::vector<std::vector<std::vector<double>>>(n, 
                std::vector<std::vector<double>>(n, 
                std::vector<double>(n, 0.0)));

            CalculateDzMap();

            std::random_device rd;  // Seed generator
            std::mt19937 gen(rd());
            std::vector<Artifact> artifacts;
            std::uniform_int_distribution<int> dist(0, n_);
            std::vector<int> pos;
            

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
            
            B0_val_ = 3.0;
            Bz_ = applyIFFT3D(multip_,n_, B0_val_);

            ALL_positions_ = GetALL_positions(n_,n_,n_,numberofprotons_, Occupied_positions_);
            ALL_positions_init_ = ALL_positions_;
            plotBzWithParticles(Bz_,Occupied_positions_, n_);
            plotBzWithParticles(ChiMap_,Occupied_positions_, n_);
        }

      

    void CalculateDzMap() {
        #pragma omp parallel for collapse(3)
        for (int x = -n_ / 2; x < n_/2; x++) {
            for (int y = -n_ / 2; y < n_/2; y++) {
                for (int z = -n_ / 2; z < n_ / 2; z++) {
                   
                    std::vector<int> position = {x, y, z};
        
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
    

    std::complex<double> SimulateDiffusionSteps(int NrOfSteps) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0,std::sqrt(2 * D_ * dt_));

      
        std::vector<std::vector<std::vector<double>>> Paths(ALL_positions_.size(), 
                                                            std::vector<std::vector<double>>(3, std::vector<double>(NrOfSteps, 0.0)));

       
        #pragma omp parallel for
        for (int i = 0; i < ALL_positions_.size(); i++) {
            std::vector<int> position = ALL_positions_[i]; 
            for (int step = 0; step < NrOfSteps; ++step) {
                
                double dx = dis(gen) * n_;
                double dy = dis(gen) * n_;
                double dz = dis(gen) * n_;
            
            
                position[0] += std::round(dx);
                position[1] += std::round(dy);
                position[2] += std::round(dz);


                position[0] = ((position[0] + n_/2) % n_ + n_) % n_ - n_/2;
                position[1] = ((position[1] + n_/2) % n_ + n_) % n_ - n_/2;
                position[2] = ((position[2] + n_/2) % n_ + n_) % n_ - n_/2;
               
                Paths[i][0][step] = position[0];
                Paths[i][1][step] = position[1];
                Paths[i][2][step] = position[2];
            }

            ALL_positions_[i] = {position[0],position[1],position[2]};
    
        }   

        
        int x,y,z;
        std::complex<double> totalPhase, phaseExp;
        double omega;
        std::vector<std::complex<double>> phases;
        totalPhase = 0; 
    
        for(int i = 0; i < numberofprotons_; i++) { 

            double phase = 0;  // Startwert 1 + 0i

            for (int j = 0; j < NrOfSteps; j++) {
                x = static_cast<int>(Paths[i][0][j] + n_ / 2);
                y = static_cast<int>(Paths[i][1][j] + n_ / 2);
                z = static_cast<int>(Paths[i][2][j] + n_ / 2);
                
                omega = GAMMA * Bz_[x][y][z];
                phase += omega * dt_;
            }

            phaseExp = std::exp(std::complex<double>(0,phase));
            phases.push_back(phaseExp);
            totalPhase += phaseExp;
            
        }

        totalPhase = totalPhase / static_cast<double>(numberofprotons_);
     
       

        return totalPhase;

    }


    std::complex<double> ComputeSignalStatic(double t) { 
        std::vector<std::complex<double>> phases;
        static int x,y,z;
        std::complex<double> phase, totalPhase;
        double omega;
        totalPhase = 0; 
     
        for(int i = 0; i < numberofprotons_; i++) { 
            x = static_cast<int>(ALL_positions_init_[i][0] + n_ / 2);
            y = static_cast<int>(ALL_positions_init_[i][1] + n_ / 2);
            z = static_cast<int>(ALL_positions_init_[i][2] + n_ / 2);

            
            omega = GAMMA * Bz_[x][y][z]; 
            phase = std::exp(std::complex<double>(0, -omega * t));
            phases.push_back(phase);
            totalPhase += phase;
            

        }
        
        return totalPhase / static_cast<double>(numberofprotons_);
    }

    

};




int main() {
    omp_set_num_threads(10); 
    int n = 300;  //grid points 
    int numberofprotons = 1e4;
    double L = 1.0f; // voxelsize
    int SIZE_arti = 10; // sizeof artifact
    double DeltaChi = 1e-9;
    int N = 100; //numberofartifatcs
    double dt = 0.001;
    


    std::vector<double> B0 = {1.0f, 0.0f, 0.0f};  // External magnetic field
    Voxel voxel(n, L, SIZE_arti, DeltaChi, N, numberofprotons, B0, dt);


    double eta = N * pow((SIZE_arti / static_cast<double>(n)),3) * M_PI * 4 /3;
    double xtot = voxel.Occupied_positions_.size() * DeltaChi;
    double R2p = (2 * M_PI) * eta * GAMMA * 3 * xtot / (9 * std::sqrt(3));

    std::cout << "Volume Fraction: " << eta  << std::endl;
    std::cout << "Voxel Suszept: " << xtot<< std::endl;
    std::vector<double> magnitudes;
    std::vector<double> times;
    std::vector<double> signal;
    std::vector<double> star;

    for(int t = 0; t < 100; t++){
        std::cout << "Timestep: " << t*dt << std::endl;
        signal.push_back(std::abs(voxel.SimulateDiffusionSteps(10)));
        magnitudes.push_back(std::abs(voxel.ComputeSignalStatic(t * dt)));
        times.push_back(t * dt);
        star.push_back(std::exp(-R2p * t * dt));
        //std::cout << voxel.ALL_positions_[0][0] << std::endl;

    }

    plt::figure_size(800, 600);
    plt::plot(times, signal);  // Blaue Linie für Simulation
    plt::plot(times, magnitudes);  // Rote gestrichelte Linie
    plt::plot(times, star);
    // Achsenbeschriftung & Titel
    plt::xlabel("Time (s)");
    plt::ylabel("Signal");
    plt::title("Signal Decay over Time");
    plt::legend();
    plt::grid(true);
    
    // Anzeige des Plots
    plt::show();
    
    
    
   

    // Initialize artifacts

    

    return 0;
}
