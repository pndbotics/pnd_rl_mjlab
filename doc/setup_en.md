# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 22.04 
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 550 or later  

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n pnd_rl_mjlab python=3.11
```

### 1.3 Activate the Virtual Environment

```bash
conda activate pnd_rl_mjlab
```

---

## 2. Installing

### 2.1 Download the Project

Clone the repository using Git:

```bash
git clone https://github.com/pndbotics/pnd_rl_mjlab.git
```

### 2.2 Install Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

#### 2.2.1 Install Mujoco-Warp

Mujoco-Warp is a GPU-optimized extension of the MuJoCo physics simulator developed by Google DeepMind.
It is designed to leverage NVIDIA GPUs for high-performance simulation and rendering.
Install it using the following command:

```bash
pip install "mujoco-warp@git+https://github.com/google-deepmind/mujoco_warp@9491175b7cbea87e28d3e3e67733095317c33398"
```
or
```bash
pip install "git+ssh://git@github.com/google-deepmind/mujoco_warp@9491175b7cbea87e28d3e3e67733095317c33398"
```

#### 2.2.2 Install Remaining Dependencies

All other dependencies are specified in the setup.py file.
Navigate to the project root directory and install them with:

```bash
cd pnd_rl_mjlab
pip install -e .
```

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.

