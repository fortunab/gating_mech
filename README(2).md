# Gating Mechanism

This repository contains Python scripts for running experiments and generating figures related to gating mechanisms, ablation analysis, activation behavior, and figure generation.

You can run either one specific experiment script or all available scripts together using `src/run\_all.py`.

\---

## 1\. Install Git

### Windows

Download and install Git for Windows:

https://git-scm.com/download/win

After installation, open a new terminal and verify Git:

```powershell
git --version
```

Expected output:

```text
git version 2.xx.x.windows.x
```

### Linux/macOS

Git is often already installed. Check with:

```bash
git --version
```

If Git is missing, install it using your system package manager.

\---

## 2\. Clone the repository

```bash
git clone https://github.com/fortunab/gating\_mech.git
cd gating\_mech
```

\---

## 3\. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

or, if `python` is not recognized:

```powershell
py -m venv .venv
```

### Linux/macOS

```bash
python3 -m venv .venv
```

\---

## 4\. Install dependencies

### Windows

```powershell
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

\---

# Running the experiments

## Option A: Run everything

Use this command to execute all available experiment scripts through `run\_all.py`.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src\\run\_all.py
```

### Linux/macOS

```bash
./.venv/bin/python src/run\_all.py
```

\---

## Option B: Run scripts independently

You can run each experiment separately.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src\\ts.py
.\\.venv\\Scripts\\python.exe src\\fig.py
.\\.venv\\Scripts\\python.exe src\\gate\_figures.py
.\\.venv\\Scripts\\python.exe src\\federated\_and\_ablation.py
.\\.venv\\Scripts\\python.exe src\\activation\_grid.py
```

### Linux/macOS

```bash
./.venv/bin/python src/ts.py
./.venv/bin/python src/fig.py
./.venv/bin/python src/gate\_figures.py
./.venv/bin/python src/federated\_and\_ablation.py
./.venv/bin/python src/activation\_grid.py
```

\---

## Optional: activate the virtual environment

Instead of writing the full `.venv` Python path every time, you can activate the virtual environment.

### Windows CMD

```cmd
.venv\\Scripts\\activate.bat
```

### Windows PowerShell

```powershell
.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks script execution, use the non-activation commands above instead.

### Linux/macOS

```bash
source .venv/bin/activate
```

After activation, you can run scripts like this:

```bash
python src/ts.py
python src/fig.py
python src/gate\_figures.py
python src/federated\_and\_ablation.py
python src/activation\_grid.py
python src/run\_all.py
```

\---

## Repository structure

```text
gating\_mech/
├── src/
│   ├── ts.py
│   ├── fig.py
│   ├── gate\_figures.py
│   ├── federated\_and\_ablation.py
│   ├── activation\_grid.py
│   └── run\_all.py
├── requirements.txt
└── README.md
```

\---

## Output

Generated results and figures are usually saved in project folders such as:

```text
results/
figures/
```

If these folders do not exist, the scripts may create them automatically, depending on the implementation.

\---

## Notes

* Use the virtual environment commands to avoid dependency conflicts with other Python projects.
* On Windows, the non-activation commands are recommended because they avoid PowerShell execution-policy issues.
* To run only one experiment, execute only the corresponding script from `src/`.
* To run the full pipeline, execute `src/run\_all.py`.

