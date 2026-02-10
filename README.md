# Vis-OCT Explorer

This repository provides software tools for processing and exploring raw visible light optical coherence tomography (vis-OCT) spectral data, with a focus on flexibility, reproducibility, and research use.

The software is designed for offline reconstruction and post-processing of acquired OCT data and is independent of specific scanning hardware implementations.

## Features
- Reconstruction from raw OCT spectral interferogram data
- Support for dual-spectrometer balanced detection reconstruction
- Automated dispersion compensation for improved reconstruction quality
- User-friendly graphical interface for biomedical OCT image analysis

---

## Installation

### Option 1: Precompiled Windows Executable (Recommended)
'Vis-OCT Explorer.exe' is provided as a precompiled package for 64-bit Windows systems.
- Download the executable package from [GitHub Releases](https://github.com/FOIL-NU/VisOCT-Explorer/releases/tag/v3.7)  
- Before first use, install the required Microsoft support package  ('Microsoft Visual C++ Redistributable, x64').  The installer is included in the downloaded folder.
- A CUDA-compatible GPU is not required, but is strongly recommendedto significantly reduce processing time.
  - GPU acceleration requires 'CUDA version 12.2'
  - The CUDA installation package is included in the downloaded folder
- Run `Vis-OCT Explorer.exe` to launch the software.

### Option 2: Build from Source

The source code of 'Vis-OCT Explorer' was developed using 'Python 3.11' and 'CUDA 12.2'.

After cloning the repository:
1. Create and activate a Python virtual environment, then install
   dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
2. Launch the software:
   ```bash
   python main.py

## Quick Start

## Example Data
Small example datasets for demonstration and testing are available here.

## How to Cite
If you use **Vis-OCT Explorer** in your research or publications, please cite the following preprint:

Fan, W., Xu, F., et al. 
*Vis-OCT Explorer: An open-source platform for processing and exploring visible-light OCT data.* bioRxiv (2025). https://www.biorxiv.org/content/10.1101/2025.10.01.679626v1

This citation will be updated once the peer-reviewed version becomes available.

## Licensing
- The source code in this repository is licensed under the BSD-3-Clause License.  
- Pre-compiled executables (`.exe`) distributed via GitHub Releases and the project website are provided free for academic and non-commercial research use only under a separate binary license.

Use of the executable to generate results for scholarly publications is permitted with proper citation.
