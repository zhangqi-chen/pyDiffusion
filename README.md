# **pyDiffusion**

**pyDiffusion** combines tools like **diffusion simulation**, **diffusion data smooth**, **forward simulation analysis (FSA)**, etc. to help people analyze diffusion data efficiently.

## Diffusion Simulation

Based on Ni-Mo interdiffusion coefficients data at 1100C, simulate the diffusion process for 800 hours. See [example](docs/examples/DiffusionSimulation.md)

![NiMo](docs/examples/DiffusionSimulation_files/DiffusionSimulation_3.png)

## Forward Simulation Analysis (FSA)

Calculate interdiffusion coefficients of Ni-Mo at 1100C based on raw diffusion data (1000 hours). See [example](docs/examples/ForwardSimulationAnalysis.md)

![NiMo_fsa](docs/examples/ForwardSimulationAnalysis_files/ForwardSimulationAnalysis_2.png)

## Error Analysis

The interdiffusion coefficients in Ti-Zr system at 1000C are calculated using FSA. The error bounds of the diffusivity data are estimated using error analysis tool. See [example](docs/examples/ErrorAnalysis.md)

![TiZr_error](docs/examples/ErrorAnalysis_files/ErrorAnalysis_3.png)