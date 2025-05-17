#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <fftw3.h>
#include <unordered_set>
#include <omp.h>
#include <ctime>
#include <random>
#include <complex>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <atomic>

#define GAMMA 267522187

 // Set the number of threads you want to use

namespace plt = matplotlibcpp;


void plotBzWithParticles(std::vector<std::vector<std::vector<double>>> Bz, 
                         std::vector<std::vector<int>>& particles, 
                         int n) {
    // Nehmen wir an, wir wollen einen 2D-Schnitt von Bz plotten (z.B. Bz[i][j])
    int slice = n/ 2 + 50; // Mittlere Schicht als Beispiel
    std::vector<std::vector<double>> Bz_slice(n, std::vector<double>(n));
   
    // Extrahiere einen 2D-Slice von Bz entlang der z-Achse (Bz[i][j][slice])
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Bz_slice[i][j] = Bz[slice][j][i]; // Slice durch die dritte Dimension
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
    //plt::title("Bz Distribution with Particle Positions");
    // Entferne Achsenticks und Labels
  
    plt::axis("off");  // Optional: entfernt auch die Achsenrahmen
    plt::save("MonteCarlo.pdf",400);
    plt::show();

}

struct TupleHash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);  // einfache Hash-Kombination
    }
};

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

std::vector<std::vector<int>> GetALL_positions(int xlen, int ylen, int zlen, int num_positions, const std::unordered_set<std::tuple<int,int,int>, TupleHash>& Occupied_positions){
    std::unordered_set<std::tuple<int,int,int>, TupleHash> positions_set;
    positions_set.reserve(num_positions + Occupied_positions.size());

    std::random_device rd;

    #pragma omp parallel
    {
        std::mt19937 thread_gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<double> dist_x(-xlen / 2 + 1, xlen / 2 - 1);
        std::uniform_real_distribution<double> dist_y(-ylen / 2 + 1, ylen / 2 - 1);
        std::uniform_real_distribution<double> dist_z(-zlen / 2 + 1, zlen / 2 - 1);

        #pragma omp for
        for (int i = 0; i < num_positions; ++i) {
            std::tuple<int,int,int> new_position;
            bool is_unique = false;

            while (!is_unique) {
                int x = static_cast<int>(std::round(dist_x(thread_gen)));
                int y = static_cast<int>(std::round(dist_y(thread_gen)));
                int z = static_cast<int>(std::round(dist_z(thread_gen)));

                new_position = std::make_tuple(x, y, z);

                if (Occupied_positions.find(new_position) == Occupied_positions.end() &&
                    positions_set.find(new_position) == positions_set.end())
                {
                    is_unique = true;
                }
            }

            #pragma omp critical
            positions_set.insert(new_position);
        }
    }

    // Jetzt in vector<vector<int>> umwandeln:
    std::vector<std::vector<int>> positions;
    positions.reserve(positions_set.size());
    for (const auto& pos : positions_set) {
        positions.push_back({std::get<0>(pos), std::get<1>(pos), std::get<2>(pos)});
    }

    return positions;
}

class Artifact {
    public:
        std::vector<int> positionmain_;
        double suscept_, totsuscept_;
        int size_;
        std::vector<std::vector<int>> positions_;

        Artifact(std::vector<int> positionmain, int pm, int size, double suscept, int n) {
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
};

std::unordered_set<std::tuple<int,int,int>, TupleHash> GetOccupiedPositions(const std::vector<Artifact>& artifacts) {
    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_set<std::tuple<int,int,int>, TupleHash>> thread_sets(num_threads);

    const size_t total = artifacts.size();
    std::atomic<size_t> processed_count(0);
    std::atomic<int> last_percent(-1); // Damit 0% auch ausgegeben wird

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_set = thread_sets[tid];

        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < total; ++i) {
            for (const auto& pos : artifacts[i].positions_) {
                if (pos.size() == 3) {
                    local_set.emplace(pos[0], pos[1], pos[2]);
                }
            }

            // Fortschritt zählen
            size_t current = ++processed_count;
            int percent = static_cast<int>((100.0 * current) / total);
            if (percent > last_percent) {
                last_percent = percent;
                #pragma omp critical
                std::cout << "\rFortschritt: " << percent << "% (" << current << "/" << total << ")" << std::flush;
            }
        }
    }

    std::cout << "\n";

    // Alles in ein Set zusammenführen
    std::unordered_set<std::tuple<int,int,int>, TupleHash> occupiedSet;
    size_t estimated_total = 0;
    for (const auto& s : thread_sets) estimated_total += s.size();
    occupiedSet.reserve(estimated_total);

    for (const auto& s : thread_sets) {
        occupiedSet.insert(s.begin(), s.end());
    }

    return occupiedSet;
}


class Proton {
    public:
        std::vector<int> position_;
        std::complex<double> phase_;
        std::vector<std::vector<int>> TrackPostitions_;
        std::vector<std::complex<double>> TrackPhases;

        Proton(const std::vector<int>& position)
        : position_(position), phase_(1.0, 0.0) {}
};

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
        double nb_ = static_cast<double>(n_);
   
        double Vm_ = 1 / (nb_ * nb_ * nb_);

        std::vector<double> B0_;
        std::vector<std::vector<std::vector<double>>> dz_;
        std::vector<std::vector<std::vector<double>>> ChiMap_;
        std::vector<std::vector<std::vector<double>>> Bz_;
        std::vector<std::vector<int>> ALL_positions_;
       
        std::unordered_set<std::tuple<int,int,int>, TupleHash> Occupied_positions_;
        std::vector<Proton> Protons_, Protons_init_;

        
        fftw_complex* dz_fftw_;
        fftw_complex* ChiMap_fftw_;
        fftw_complex* Bz_fftw_;

        Voxel(int n, double L, int SIZE_arti, double DeltaChi, int N, int numberofprotons, std::vector<double> B0, double dt): 
            n_(n), L_(L), SIZE_arti_(SIZE_arti), DeltaChi_(DeltaChi), N_(N),numberofprotons_(numberofprotons), B0_(B0), dt_(dt) {

            std::cout << "\nSIMULATOR\n\nInitialize Voxel..." << std::endl;

            D_ = 4e-1; 
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
            
            B0_val_ = 3.0;
            Bz_ = applyIFFT3D(multip_,n_, B0_val_);
            
            std::cout << "Initialize Protons..." << std::endl;

            ALL_positions_ = GetALL_positions(n_,n_,n_,numberofprotons_, Occupied_positions_);

            for (int i = 0; i < numberofprotons_; i++) {
                Protons_.push_back(Proton(ALL_positions_[i]));
            }

            Protons_init_ = Protons_;

            shift3DArray(Bz_, n_);
            //plotBzWithParticles(Bz_,ALL_positions_, n_);
            //plotBzWithParticles(ChiMap_,ALL_positions_, n_);
            std::cout << "\nVoxel Initialization Done\n" << std::endl;

            
        }
  

    void SaveEveryEntry(std::vector<std::vector<std::vector<double>>> array) {
            // Get the dimensions of the array
            int x = array.size();
            int y = array[0].size();
            int z = array[0][0].size();
        
            // Flatten the 3D array into a 1D vector
            std::vector<double> flattened;
        
            for (int i = 0; i < x; ++i) {
                for (int j = 0; j < y; ++j) {
                    for (int k = 0; k < z; ++k) {
                        flattened.push_back(array[i][j][k]);
                    }
                }
            }
        
            // Size of the flattened array
            size_t N = flattened.size();
        
            // Create FFTW arrays
            fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
            fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        
            // Copy flattened data into FFTW input array (real part only, imaginary part is set to 0)
            for (size_t i = 0; i < N; ++i) {
                in[i][0] = flattened[i];  // Real part
                in[i][1] = 0;             // Imaginary part
            }
        
            // Perform the FFT using FFTW
            fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(p);
        
            // Prepare the frequency and magnitude for plotting
            std::vector<double> frequencies(N);
            std::vector<double> magnitude(N);
        
            for (size_t i = 0; i < N; ++i) {
                // Frequency
                frequencies[i] = static_cast<double>(i);
                
                // Magnitude (square root of the sum of squares of real and imaginary parts)
                magnitude[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
            }
            
            std::rotate(frequencies.begin(), frequencies.begin() + N / 2, frequencies.end());
            std::rotate(magnitude.begin(), magnitude.begin() + N / 2, magnitude.end());
        
            // Plot the FFT result
            plt::figure_size(800, 600);
            plt::plot(frequencies, magnitude);
            plt::title("Fourier Transform of Flattened Array");
            plt::xlabel("Frequency");
            plt::ylabel("Magnitude");
            plt::show();
        
            // Clean up FFTW resources
            fftw_destroy_plan(p);
            fftw_free(in);
            fftw_free(out);
        }
    
    void CalculateDzMap() {
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

    std::complex<double> SimulateDiffusionSteps(int NrOfSteps, double t) {
       
        std::random_device rd;
        std::mt19937 gen(rd());

        double dtM_ = dt_ / static_cast<double>(NrOfSteps);

        std::normal_distribution<> dis(0,std::sqrt(2 * D_ * dtM_));

      
        std::vector<std::vector<std::vector<double>>> Paths(numberofprotons_, 
                                                            std::vector<std::vector<double>>(3, std::vector<double>(NrOfSteps, 0.0)));

         
        #pragma omp parallel for
        for (size_t i = 0; i < numberofprotons_; i++) {
            std::vector<int> position = Protons_[i].position_;
            std::tuple<int,int,int> candidate;

            for (int step = 0; step < NrOfSteps; ++step) {
                bool collision;
                
                do {
                    double dx = dis(gen) * n_;
                    double dy = dis(gen) * n_;
                    double dz = dis(gen) * n_;
        
                    int newX = position[0] + std::round(dx);
                    int newY = position[1] + std::round(dy);
                    int newZ = position[2] + std::round(dz);
        
                    // Periodic boundary conditions
                    newX = ((newX + n_/2) % n_ + n_) % n_ - n_/2;
                    newY = ((newY + n_/2) % n_ + n_) % n_ - n_/2;
                    newZ = ((newZ + n_/2) % n_ + n_) % n_ - n_/2;
        
                    candidate = std::make_tuple(newX, newY, newZ);
        
                    // Prüfen, ob Position besetzt
                    collision =  (Occupied_positions_.find(candidate) != Occupied_positions_.end());
        
                    // Wenn collision == true, wird while nochmal ausgeführt und neuer Schritt probiert
                } while (collision);
        
                // freie Position gefunden -> setzen
                position[0] = std::get<0>(candidate);
                position[1] = std::get<1>(candidate);
                position[2] = std::get<2>(candidate);
              
        
                Paths[i][0][step] = position[0];
                Paths[i][1][step] = position[1];
                Paths[i][2][step] = position[2];
            }
        
            // Partikelposition aktualisieren
            Protons_[i].position_ = {position[0], position[1], position[2]};
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

        std::vector<int> position = ALL_positions_[20];
        
        std::cout << "Position:  " << position[0] << " " << position[1] << " " << position[2] << std::endl;
        totalPhase = totalPhase / static_cast<double>(numberofprotons_);
     
        return totalPhase;

    }

    std::complex<double> ComputeSignalStatic(double t) { 
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


};


int main() {
    omp_set_num_threads(20); 
    int n = 500;  //grid points 
    int numberofprotons = 1e4;
    double L = 1.0f; // voxelsize
    int SIZE_arti = 10; // sizeof artifact
    double DeltaChi =5e-15;
    int N = 2000; //numberfartifatcs
    double dt = 0.0001;
    double nd = static_cast<double>(n);
    double Vm = 1 / (nd * nd * nd);
    

    std::vector<double> B0 = {0.0f, 0.0f, 3.0f};  // External magnetic field
    Voxel voxel(n, L, SIZE_arti, DeltaChi, N, numberofprotons, B0, dt);


    double eta = voxel.Occupied_positions_.size() * Vm;
    double xtot = voxel.Occupied_positions_.size() * DeltaChi;  
    double R2p = (8 * M_PI*M_PI) * eta * GAMMA * 3 * xtot / (9 * std::sqrt(3));

    std::cout << "Volume Fraction: " << eta  << std::endl;
    std::cout << "Voxel Suszept: " << xtot<< std::endl;
    std::cout << "R2p " << R2p << std::endl;
    std::vector<double> magnitudes;
    std::vector<double> times;
    std::vector<double> signal;
    std::vector<double> star;
    double inter, stat;

        
    for(int t = 0; t < 1000; t++){
        inter = std::abs(voxel.SimulateDiffusionSteps(10, t * dt));
        stat = std::abs(voxel.ComputeSignalStatic(t * dt));

        std::cout << "Timestep: " << t*dt << "  Diffusion: " << inter << " Static Dephasing: " << stat << "  Analytic: " << std::exp(- R2p * t * dt) << std::endl;

        signal.push_back(inter);
        magnitudes.push_back(stat);
        times.push_back(t * dt);
        star.push_back(std::exp(- R2p * t * dt));
    }

    //voxel.SaveEveryEntry(voxel.Bz_);
    plotBzWithParticles(voxel.Bz_, voxel.ALL_positions_, voxel.n_);

    plt::plot(times, magnitudes);
    plt::plot(times, signal);
    plt::plot(times, star);
    plt::xlabel("Time");
    plt::ylabel("Magnitude");
    plt::title("Signal decay over time");
 
    plt::show();


    return 0;
}
