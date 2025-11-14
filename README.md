# Benchmarking Universal Machine Learning Interatomic Potentials for Real-Time Analysis of Inelastic Neutron Scattering Data

Calculate phonon properties with multiple universal machine learning interatomic potentials (uMLIPs).

### Installation

To use the package, conda environment is recommended. As the required packages of the uMLIPs differ drastically, we use several conda environments to compute the force constants with different uMLIPs. The suggested environments files are included in the package. To create the environment, you can use the following command:

```bash
git clone https://github.com/maplewen4/phonon_uMLIP.git
cd phonon_uMLIP
conda env create -f env_phonon_uMLIPXXXXX.yml
```

where XXXXX stands for the indexes of the uMLIPs. Specifically, `1-5` are continuous, meaning 1, 2, 3, 4, and 5, while other numbers are discrete. The uMLIPs and their corresponding indexes are listed below:

|      uMLIP       | Index |
| :--------------: | :---: |
|   eSEN-30M-OAM   |  12   |
|      ORB v3      |  11   |
| SevenNet-MF-ompa |   9   |
|   GRACE-2L-OAM   |  10   |
|   MatterSim 5M   |   5   |
|    MACE-MPA-0    |   8   |
|      eqV2 M      |   6   |
|      ORB v1      |   3   |
|    SevenNet-0    |   4   |
|    MACE-MP-0     |   0   |
|      CHGNet      |   1   |
|      M3GNet      |   2   |
|   MatterSim 1M   |   7   |

### Usage

To generate the force constants with the uMLIPs, you may use the associated dataset, https://doi.org/10.5281/zenodo.15298435. To use a specific uMLIP, you can use the Python file with the corresponding index. Some outperforming uMLIPs are implemented in INSPIRED, a graphic user interface, https://github.com/neutrons/inspired. 

## Citation

Bowen Han, Yongqiang Cheng, [Benchmarking universal machine learning interatomic potentials for rapid analysis of inelastic neutron scattering data](https://iopscience.iop.org/article/10.1088/2632-2153/adfa68), Machine Learning: Science and Technology, 6, 030504 (2025).

