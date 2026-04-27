# Monte Carlo Simulation of magnetized Spheres

This repository contains a Monte Carlo simulation framework for studying diffusion and transverse relaxation in heterogeneous media composed of spherical inclusions with susceptibility contrast.

The simulation models proton motion, local magnetic field perturbations, and resulting signal decay.

---

## Description

The framework simulates:

* Diffusion of protons using a Monte Carlo approach
* Local magnetic field perturbations caused by susceptibility differences
* Transverse relaxation emerging from diffusion in inhomogeneous fields

The core implementation is written in C++ for computational efficiency.

---

## Compilation

The project uses CMake and requires a C++17-compatible compiler.

### Dependencies

* OpenMP
* FFTW3
* NetCDF (C and C++ interface)
* nlohmann_json
* NIfTI library
* ZLIB

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

This produces the executable:

```bash
./simulation
```

---

## Input Configuration

The simulation is controlled via a JSON configuration file:

```json
{
  "B0": [0, 0, 3],
  "D": 2e-09,
  "DiffSteps": 10.0,
  "dt": 0.001,
  "n": 500,
  "numberofprotons": 1000000.0,
  "sigma": 0.0,
  "tsteps": 100,
  "R2": 0,
  "eta": 0.2,
  "Xtot": 2.5132741228718345e-07,
  "index": 2019,
  "mu": 1.93,
  "L": 0.000345
}
```

---

## Parameter Description

### Magnetic Field

* **$\mathbf{B}_0$ (B0)**: External magnetic field vector (Tesla)

---

### Diffusion

* **$D$**: Diffusion coefficient (m²/s)
* **dt**: Time step
* **tsteps**: Number of time steps
* **DiffSteps**: Number of substeps between stored positions (sub-diffusion steps)

---

### Simulation Domain

* **n**: Grid size ($n \times n \times n$)
* **$L$**: Physical domain size

The domain size $L$ is internally adjusted such that the mean sphere radius corresponds to at least approximately 10 voxels, ensuring sufficient spatial resolution.

---

### Proton Sampling

* **numberofprotons**: Number of simulated spins

---

### Microstructure

* **$\xi^S$ (eta)**: Volume fraction occupied by spheres
* **$\mu, \sigma$ (mu, sigma)**: Parameters of the log-normal distribution of sphere radii

The sphere radii follow a log-normal distribution:

$$
p(R) = \frac{1}{R , \sigma \sqrt{2\pi}} \exp\left( -\frac{(\ln R - \mu)^2}{2\sigma^2} \right)
$$

where:

* $\mu$ is the mean of the logarithmic radius
* $\sigma$ is the standard deviation of the logarithmic radius

---

### Susceptibility

* **$\bar{\chi}$ (Xtot)**: Bulk susceptibility of the spherical inclusions

---

### Relaxation

* **$R_{2,\mathrm{nano}}$ (R2)**: Molecular (intrinsic) transverse relaxation rate

---

### Miscellaneous

* **index**: Simulation identifier

---

## Output

Simulation results are stored in NetCDF format.

### simulation.nc

Contains:

* Proton trajectories (positions over time)
* Time vector
* Precomputed transverse signal decays under different models

The following signal representations are included:

---

**Analytical**

$$
S(t) = \exp\left( - (R_2 + R_2'), t \right)
$$

---

**Static Dephasing (no diffusion)**

Signal decay assuming static spins in the susceptibility-induced field.

---

**Diffusion (Gradient Echo / FID)**

$$
S_{\mathrm{Diffusion}}(t) = \left\langle \exp \left( i \int_0^{t} \Delta \omega(\mathbf{r}(t')) , dt' \right) \right\rangle
$$

Monte Carlo simulation of the gradient echo (free induction decay, FID) signal including diffusion, without refocusing.

---
**Second-order cumulant ($\kappa_2$)**

$$
S(t) =  e^{-\frac{1}{2} \kappa_2(t)} 
$$

Approximation using only the second cumulant of the phase distribution.

---

**Fourth-order cumulant ($\kappa_2, \kappa_4$)**

$$
S(t) =  e^{- \frac{1}{2}\kappa_2(t) + \frac{1}{24}\kappa_4(t)} 
$$

Extension including the fourth cumulant of the phase distribution.

---

**SpinEcho**

$$
S_{\mathrm{SpinEcho}}(t) = \left\langle \exp \left( i \left[ \int_0^{t/2} \Delta \omega(\mathbf{r}(t')) dt' - \int_{t/2}^{t} \Delta \omega(\mathbf{r}(t')) dt' \right] \right) \right\rangle
$$

Monte Carlo simulation of the spin-echo signal, including refocusing at $t = \mathrm{TE}/2$.

---

### Maps.nc

Contains:

* Susceptibility maps
* Microstructure-related fields
* Derived spatial quantities

---

Both files allow efficient post-processing and reproducible analysis of large-scale simulations.

---

## Usage

```bash
./simulation
```

---

## Notes

The simulation captures microstructural effects arising from the interaction of diffusion and susceptibility-induced field variations. Transverse relaxation emerges naturally from this interaction.

---

## License

MIT License
