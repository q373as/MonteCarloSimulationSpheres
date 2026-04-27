#ifndef SRC_HPP
#define SRC_HPP

#include <vector>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <set>

/**
 * @brief Represents a susceptibility-inducing artifact within the voxel space.
 */
class Artifact {
    public:
        std::vector<int> positionmain_;                     ///< Main position of the artifact (voxel indices)
        double suspV_;                                    ///< Susceptibility value of the artifact
        double totsuscept_;                                 ///< Total susceptibility including volume effects
        int size_;    
        double Vm_;  
        double etai_;                                   ///< Size of the artifact (e.g. sphere radius in voxels) 
        std::vector<std::vector<int>> positions_;           ///< All occupied positions by the artifact

        /**
         * @brief Constructs an Artifact.
         * @param positionmain Main voxel index
         * @param pm Padding or position mode
         * @param size Radius or extent of the artifact
         * @param suscept Base susceptibility
         * @param n Grid size (for boundary checking)
         */
        Artifact(std::vector<int> positionmain, int pm, int size, int n);
};

/**
 * @brief Represents a single diffusing proton within the voxel grid.
 */
class Proton {
    public:
        std::vector<double> initial_position_;
        std::vector<double> position_;                      ///< Continuous position in voxel coordinates
        std::complex<double> phase_;                        ///< Complex-valued phase of the proton
        std::vector<std::vector<double>> TrackPostitions_;  ///< History of positions during diffusion
        std::vector<std::complex<double>> TrackPhases_;      ///< History of phases during diffusion
        /**
         * @brief Constructs a Proton at a given position.
         * @param position 3D position in voxel coordinates
         */
        Proton(const std::vector<double>& position);
};

struct SignalResults {
    std::complex<double> signal_kappa_2;
    std::complex<double> signal_kappa_4;
    std::vector<double> moments;         
    std::vector<double> cumulants;        
    std::complex<double> totalPhase;      
};


/**
 * @brief Represents the simulation domain (a voxel grid) and manages diffusion, susceptibility and signal.
 */
class Voxel {
    public:
        int n_;                                       ///< Grid size per dimension (n x n x n)
        double L_;                                    ///< Physical voxel size in m)
        int N_ = 0;                                   ///< Total number Artifacts. NULL per default
        double mu_, sigma_;                           ///< Parameters for distribution of susceptibility Artifacts
        double Xtot_;                                 ///< Total Susceptiblity of susceptibility sources
        int numberofprotons_;                         ///< Number of protons in the simulation
        
        double dt_;                                   ///< Time step size for readout
        double D_;                                    ///< Diffusion coefficient
        double eta_, ratio_;                                  ///< Volumee fraction of artifacts          ///< Effective susceptibility

        double nb_ = static_cast<double>(n_);         ///< Grid size as double
        double Vm_ = 1 / (nb_ * nb_ * nb_);           ///< Voxel volume (assuming unit cube)

        std::vector<double> B0_;                      ///< Background magnetic field direction (3D)
        std::vector<std::vector<std::vector<double>>> ChiMap_;   ///< Susceptibility map
        std::vector<std::vector<std::vector<double>>> Bz_;  
        std::vector<std::vector<std::vector<std::complex<double>>>> Dk_;      ///< Field map (Bz) after convolution
        std::vector<std::vector<double>> ALL_positions_;         ///< All possible proton starting positions
        std::set<std::vector<int>> Occupied_positions_;          ///< Voxel indices already occupied by artifacts
        std::vector<Artifact> artifacts;                         ///< List of all placed artifacts
        std::vector<Proton> Protons_;                            ///< Active proton list (updated during simulation)
        

        fftw_complex* dz_fftw_;                 ///< FFTW array: dipole kernel
        fftw_complex* ChiMap_fftw_;             ///< FFTW array: susceptibility map
        fftw_complex* Bz_fftw_;                 ///< FFTW array: resulting Bz field after convolution

        double B0_val_ = sqrt(B0_[0]*B0_[0] + B0_[1]*B0_[1] + B0_[2]*B0_[2]); ///< Magnitude of B0

        /**
         * @brief Constructs the voxel grid and initializes simulation parameters.
         * @param n Grid size (n x n x n)
         * @param L Physical size per voxel
         * @param Xtot Total susceptibility fraction
         * @param eta Susceptibility-to-field scaling
         * @param numberofprotons Number of protons to simulate
         * @param B0 Background field vector
         * @param D Diffusion coefficient
         * @param dt Time step size
         * @param mu Mean susceptibility
         * @param sigma Susceptibility standard deviation
         */
        Voxel(int n, double L, double Xtot, double eta, int numberofprotons,
            std::vector<double> B0, double D, double dt, double mu, double sigma, double ratio);

        /**
         * @brief Saves all entries of a 3D array (for debugging or export).
         * @param array 3D array to process
         */
        void SaveEveryEntry(std::vector<std::vector<std::vector<double>>> array);

        /**
         * @brief Computes the dipole kernel dz in the spatial domain.
         */
    
        double interpolateBz(double x, double y, double z);

        /**
         * @brief Simulates the diffusion of protons over multiple steps.
         * @param NrOfSteps Number of diffusion steps
         * @param t Total simulation time
         * @return Complex-valued MR signal
         */
        SignalResults SimulateDiffusionSteps(int NrOfSteps);

        /**
         * @brief Computes the static signal (without diffusion).
         * @param t Echo time
         * @return Complex-valued MR signal
         */
        std::complex<double> ComputeSignalStatic(double t);

        std::vector<double> ComputeTemporalACF(int tsteps);

        std::complex<double> SimulateSpinEchoSignal(int diffusionSteps, double dt, double TE);

};
/**
 * @brief Returns all occupied voxel positions from a list of artifacts.
 * @param artifacts Vector of Artifact objects
 * @return Set of 3D voxel indices
 */
std::set<std::vector<int>> GetOccupiedPositions(const std::vector<Artifact>& artifacts);

/**
 * @brief Generates a list of random positions in the voxel grid, avoiding occupied voxels.
 * @param xlen Grid size X
 * @param ylen Grid size Y
 * @param zlen Grid size Z
 * @param num_positions Number of desired positions
 * @param Occupied_positions Set of voxel indices to avoid
 * @return List of random unoccupied positions
 */
std::vector<std::vector<double>> GetALL_positions(int xlen, int ylen, int zlen,
                                                  int num_positions,
                                                  const std::set<std::vector<int>>& Occupied_positions);





#endif // SRC_HPP
