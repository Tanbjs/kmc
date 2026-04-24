# Data-driven for autonomous underwater vehicle 

## Overview
The project aims to simulate and compare various data-drive techniques on the autonomous underwater vehicle. Such techniques include:
- System identification using Koopman operator theory
    - Dynamic Mode Decomposition wtih control (DMDc)
    - Extended Dynamic Mode Decomposition with control (EDMDc)
    - Deep koopman with control (DKC)
    - Deep koopman with control using physics-informed neural network (DKC-PINN)

## Repository Structure
- `data/`: Contains scripts for raw, preprocessed, featured data.
- `config/`: Configuration files for different models and experiments.
- `sysid/`: library of various system identification techniques.
- `notebooks/`: Jupyter notebooks for prototype.
- `src/`: Main programs.
- `tools/`: Utility scripts for various tasks.

## Prerequisites
- Python 3.10.13
- Git
- data/raw (must be created manually) at the root of the project.

## Getting Started
1. Clone the repository:
    ```bash 
    git clone https://github.com/Tanbjs/xplorer_mini_sysid.git 
    ```

2. Navigate to the project directory:
    ```bash 
    cd xplorer_mini_sysid
    ```

3. Create directory (Make sure to place your raw data files in this directory):
    ```bash
    mkdir data/raw
    ```

3. Install the required packages (Optional: activate a virtual environment then run the following commands):
    ```bash
    pip install -r requirements/base.txt
    pip install -r requirements/dev.txt
    ```

4. Run the local experiment
    - Linux or macOS
    ```bash
   source tools/run/run_local.sh
    ```
    - Docker
    ```bash
   docker build -t <image_name> .
   docker run --gpus all -it --rm -v $(pwd):/workspace <image_name>
   source tools/run/run_local.sh
    ```