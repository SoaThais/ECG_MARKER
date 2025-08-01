# ECG_MARKER

## Install using PyPi

1. Install System Requirements

```bash
sudo apt install python3-pip

sudo apt install python3-tk

sudo apt-get install python3-pil.imagetk
```

2. Install the latest version available on Pypi: https://pypi.org/project/ecg-marker/

```bash
pip install ecg-marker==x.x.x
```

### Test Commands

1. Download the test files from the repository: https://github.com/SoaThais/ECG_MARKER/tree/main

2. For unprocessed files

```bash
python3 python3 -m ecg_marker -i ./input/ -f 0 
```

3. For processed files 

```bash
python3 python3 -m ecg_marker  -i ./output/ecg_data.txt -r 0 
```

## Install source code on Windows

1. Enable WSL

Open PowerShell as administrator and run the following command to install WSL:

```bash
wsl --install
```

2. Install Ubuntu on WSL

After enabling WSL, open the Microsoft Store, search for Ubuntu (or another Linux distribution you prefer), and install it.

3. Update packages in Ubuntu

In the WSL terminal, run the following commands to update the system packages:

```bash
sudo apt update
sudo apt upgrade
```

4. Install Miniconda: 
    
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

```bash
source ~/.bashrc
```

5. Install and configure Git

```bash
sudo apt install git
```

```bash
git config --global user.name "your_name"
```

```bash
git config --global user.email "your_email@example.com"
```

6. Clone the repository and navigate to the project directory:

```bash
git clone git@github.com:SoaThais/ECG_MARKER.git
```

```bash
cd ECG_MARKER/
```

7. Create a Conda environment with the project dependencies

```bash
conda env create -f environment.yml
```

8. Activate the Conda environment

```bash
conda activate ecg_marker_env
```

9. Install the library 

```bash
conda install -c conda-forge libxcb
```

### Test Commands

1. For unprocessed files

```bash
python3 ./src/ecg_marker/ecg_marker.py  -i ./input/ -f 0 
```

2. For processed files 

```bash
python3 ./src/ecg_marker/ecg_marker.py  -i ./output/ecg_data.txt -r 0 
```

## Note

If a folder is used as input, name the files in the directory in alphabetical order.

## Command line arguments

```bash
  -h, --help            show this help message and exit

  -i INPUT              Input

  -f INPUT_FILE         Input File (1) or Input Directory (0)

  -d OUTPUT_DIR         Output Directory

  -o OUTPUT_FILE        Output File

  --qrs_file QRS_FILE   Output file with QRS data

  --qt_file QT_FILE     Output file with QT data

  --vel_file VEL_FILE   Output file with estimated normalized velocity data

  --arrhythmia_file ARRHYTHMIA_FILE Output file with arrhythmia marking

  --extrasystole_file EXTRASYSTOLE_FILE Output file with extrasystole marking

  --apd_file APD_FILE   Output file with estimated APD data

  -r RAW_DATA           Raw Data (1) or not (0)
```