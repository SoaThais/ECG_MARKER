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

## Neural automatic marking (ecg_nn)

Automatic QRS marking runs on a neural model (`ecg_nn/`) instead of `neurokit2`.
Click **Automatic Marking** in the GUI, or the **ecg_nn ⚙** button (next to the
plot toolbar) to change how it behaves:

- **Noisy beat handling** -- what to do with a beat whose window overlaps a
  neighbor's: *Recovery* (rescue with a shifted/truncated window, same as
  training), *Exclude* (skip it), or *Force* (predict on it anyway, flag
  low-confidence results instead of hiding them).
- **Model** -- which production ensemble to use: *4-fold* (32 members, 4
  leave-one-out folds x 8 seeds, has real held-out validation) or *Complete*
  (64 members trained on all patients, no holdout possible).

### Model weights (`models/`)

```
models/
├── v6_daint/     legacy fallback: a single mid-training checkpoint
└── production/   current: the production QRS ensemble bundle
```

`ecg_nn` picks the best one available at runtime: the production ensemble in
`models/production/` if present, else the single checkpoint in
`models/v6_daint/`, else the original FiLM head. See
[`models/production/README.md`](models/production/README.md) for the
ensemble's full input/output contract, golden-test usage, and caveats
(notably: TF32 must stay disabled on CUDA, and `encoder_sha256` is not yet
filled in, so HuBERT-encoder compatibility isn't verified automatically).

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