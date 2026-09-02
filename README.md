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

1. Create a configuration file (see [Configuration file](#configuration-file) below) and run:

```bash
python3 ./src/ecg_marker/ecg_marker.py -c ./config.ini
```

## Configuration file

All input, output, and marking settings are now read from a single `.ini`
configuration file, passed via `-c`. Example:

```ini
[electrodes]
# List of all channel (electrode) names expected in the input files
head_file = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'HISp', 'HISd', 'VD p', 'VD 78', 'VD 56', 'VD 34', 'VD d']
# Subset of channels to be processed
head = ['VD d', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

[data]
# Input directory or file
input = ./input/
# Input File (1) or Input Directory (0)
input_file = 0
# Output directory
output_dir = ./output/
# Raw Data (1) or Processed Data (0)
raw_data = 1

[marking]
# Clean signal (1) or not (0)
clean_signal = 0
# Read simulated ECG from MonoAlg3D (1) or real ECG from the electrophysiological study (0)
ecg_mono = 0
# Vertical offset between electrode signals in the visualization window
offset = 1000
# Uncertainty in marking (ms)
uncertainty = 15
```

To reopen a previously saved session instead of raw signal files, point
`input` at the saved output file and set `raw_data = 0` and `input_file = 1`:

```ini
[data]
input = ./output/ecg_data.txt
input_file = 1
output_dir = ./output/
raw_data = 0
```

## Neural automatic marking (ecg_nn)

Trained in [ybwerneck/QRS_Detector](https://github.com/ybwerneck/QRS_Detector/tree/main).

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
└── production/   the production QRS ensemble bundle
```

`ecg_nn` uses the production ensemble in `models/production/`. See
[`models/production/README.md`](models/production/README.md) for the
ensemble's full input/output contract, golden-test usage, and caveats
(notably: TF32 must stay disabled on CUDA, and `encoder_sha256` is not yet
filled in, so HuBERT-encoder compatibility isn't verified automatically).

## Note

If a folder is used as input, name the files in the directory in alphabetical order.

## Command line arguments

```bash
  -h, --help    show this help message and exit

  -c CONFIG     Path to the configuration file (required)
```