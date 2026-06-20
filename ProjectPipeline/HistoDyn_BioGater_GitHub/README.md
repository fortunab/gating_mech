# HistoDyn::BioGater

Biologically Interpretable Dynamic Gating Architecture for Computational Pathology.

This repository contains a reproducible TensorFlow implementation of **HistoDyn::BioGater**, a lightweight dynamic-gating model for colorectal histology classification. The model replaces dense always-on computation with interpretable cheap operations such as normalization, contrast amplification, polarity mapping, binarization, and texture shifts. A learnable gate controls which operations are emphasized.

The associated paper describes BioGater as a dynamically sparse architecture for computational pathology, using biologically grounded operations regulated by top-k learning gates and L1 sparsity. It reports improvements over RandomOp Mixer, ConvMixer, and pruned baselines in accuracy, calibration, and energy efficiency. For better results, use 40 or 50 epochs, instead of 10 for each block layer. 

\---

## Repository structure

```text
HistoDyn_BioGater_GitHub/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── configs/
│   └── default.yaml
├── src/
│   └── biogater/
│       ├── data.py
│       ├── metrics.py
│       ├── models.py
│       └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_gates.py
│   └── run_all.py
├── notebooks/
└── docs/
```

\---

## Required Python version

Recommended and tested version:

```text
Python 3.10
```

Python 3.10 is recommended because TensorFlow compatibility is sensitive to Python versions. Avoid using the newest Python release unless TensorFlow explicitly supports it.

\---

## Option A: Virtual environment with venv

### 1\. Create a virtual environment

```bash
python -m venv .venv
```

This creates a separate project-specific Python environment.

### 2\. Activate the virtual environment

Windows PowerShell:

```bash
.venv/Scripts/Activate.ps1
```

Windows CMD:

```bash
.venv/Scripts/activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

This ensures that packages are installed inside the project environment, not into the current system Python installation.

### 3\. Install packages

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs TensorFlow, TensorFlow Datasets, NumPy, Matplotlib, scikit-learn, Jupyter, and YAML support.

For Python 3.12 + Tensorflow 2.20, use:
```bash
pip uninstall -y protobuf tensorflow tensorflow-intel tensorflow-datasets tensorflow-metadata tensorboard ml-dtypes

pip install tensorflow==2.20.0
pip install protobuf==6.31.1
pip install tensorflow-datasets==4.9.7
pip install tensorflow-metadata==1.17.2
pip install tensorboard==2.20.0
pip install ml-dtypes==0.5.4
```

### 4\. Run the required commands

Train BioGater:

```bash
python scripts/train.py --model biogater
```

Train ConvMixer baseline:

```bash
python scripts/train.py --model convmixer
```

Train depthwise-only baseline:

```bash
python scripts/train.py --model depthwise
```

Run all experiments:

```bash
python scripts/run_all.py
```

Plot learned gate values:

```bash
python scripts/plot_gates.py
```

### 5\. Deactivate the environment

```bash
deactivate
```

This returns the terminal to the global Python environment.

\---

## Option B: Conda environment

The Conda mechanism is similar: create a separate environment, activate it, install dependencies, run commands, then deactivate.

### 1\. Create environment

```bash
conda create -n histodyn-biogater python=3.10 -y
```

or reuse the environment file:

```bash
conda env create -f environment.yml
```

### 2\. Activate environment

```bash
conda activate histodyn-biogater
```

### 3\. Install packages

```bash
pip install -r requirements.txt
```

### 4\. Run commands

```bash
python scripts/train.py --model biogater
```

### 5\. Deactivate environment

```bash
conda deactivate
```

\---

Default split used in the scripts:

```text
4000 train
500 validation
500 test
```

Image size:

```text
75 × 75 × 3
```

Number of classes:

```text
8
```

\---

## Reproducibility workflow

Minimal workflow:

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/train.py --model biogater
python scripts/train.py --model convmixer
python scripts/plot_gates.py
deactivate
```

For Linux/macOS, replace the activation command with:

```bash
source .venv/bin/activate
```

\---

## Configuration

Default parameters are stored in:

```text
configs/default.yaml
```

Important values:

```yaml
seed: 42
img_size: 75
num_classes: 8
patch_size: 5
embed_dim: 64
num_blocks: 2
top_k: 4
l1_lambda: 0.01
batch_size: 64
epochs: 10
learning_rate: 0.001
```

To change training settings, edit this file or pass another config:

```bash
python scripts/train.py --config configs/default.yaml --model biogater
```

\---

## Output files

Training produces:

```text
outputs/models/biogater/best_model.keras
outputs/models/biogater/results.json
```

Gate visualization produces:

```text
outputs/figures/gates.png
```

\---

## Models included

### HistoDyn::BioGater

```bash
python scripts/train.py --model biogater
```

Uses a dynamic operation bank with learnable gates.

### ConvMixer

```bash
python scripts/train.py --model convmixer
```

Baseline dense lightweight mixer.

### Depthwise-only baseline

```bash
python scripts/train.py --model depthwise
```

Ablation-style lightweight baseline.

\---

## Interpretability

BioGater exposes operation gates from the dynamic cheap-operation bank. These gates can be interpreted as concept-level indicators:

```text
std_norm       -> stain / illumination normalization
scale_large    -> contrast amplification
sign           -> edge / polarity behavior
binarize       -> morphological discretization
mean_subtract  -> statistical centering
abs            -> magnitude extraction
reverse        -> texture-order shift
```

The script:

```bash
python scripts/plot_gates.py
```

generates a figure with learned gate values.

### Deactivate venv
```powershell
deactivate
```



