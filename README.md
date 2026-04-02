# Model Alignment Lab

Fine-tuning small language models with LoRA for structured outputs.

## Overview

Model Alignment Lab is a hands-on workshop focused on practical language model alignment. The goal is to take a base language model and adapt its behavior so that it follows instructions more reliably and produces structured, machine-readable outputs such as JSON.

This workshop is designed to help you understand the gap between a general-purpose base model and a model that has been tuned for a specific task format. Rather than retraining a model from scratch, we use parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation) to shape behavior with far lower compute and memory requirements.

By the end of the workshop, you will be able to:

- understand what alignment means in a practical fine-tuning workflow
- prepare instruction and response data for supervised fine-tuning
- fine-tune a model with LoRA
- evaluate behavior before and after tuning
- build a reusable workflow for future alignment experiments

## What We Are Building

In this lab, we start with a pretrained language model and adapt it to produce cleaner, more predictable outputs for a narrow task.

A typical example is moving from a loosely formatted natural-language answer to a structured response.

### Before alignment

```text
Sure! You can plan a trip by first choosing a destination and then deciding how long you want to stay.
```

### After alignment

```json
{
  "action": "plan_trip",
  "destination": "Hawaii",
  "duration_days": 5
}
```

The exact task may vary depending on the notebook or dataset, but the core idea stays the same: align the model so that its outputs are more useful for downstream systems.

## Key Concepts Covered

- alignment vs. pretraining
- supervised fine-tuning
- LoRA and low-rank updates
- dataset design for behavioral control
- structured outputs and formatting constraints
- overfitting and evaluation


## Environment Setup

You can set up the project with either `uv` or `pip`.

### Python version

This project currently targets:

- Python 3.11

On Windows ARM systems, prefer an x64 Python installation if you plan to install PyTorch and related training dependencies.

---

## Option 1: Setup with uv

This is the preferred workflow if you are using `uv` for dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/DerrickJ1612/model-alignment-lab.git
cd model-alignment-lab
```

### 2. Create a virtual environment

If you are using the default `.venv` naming convention:

```bash
uv venv --python 3.11
```

If you want to force a specific interpreter:

```bash
uv venv --python "C:\path\to\python.exe"
```

### 3. Activate the environment

Windows:

```powershell
.\venv_x86\Scripts\Activate.ps1
```

or, if using the default environment name:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install base dependencies

If using an activated custom environment such as `venv_x86`:

```bash
uv sync --active
```

If using the default `.venv`:

```bash
uv sync
```

### 5. Install training dependencies


If using `uv` with an active custom environment:

```bash
uv pip install -r requirements-train.txt
```

---

## Option 2: Setup with pip

### 1. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

macOS / Linux:

```bash
python3 -m venv venv
```

### 2. Activate the environment

Windows:

```powershell
venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 3. Install base dependencies

```bash
pip install -r requirements.txt
```

### 4. Install training dependencies

```bash
pip install -r requirements-train.txt
```

## Dependency Files

The repository uses two dependency files:

### `requirements.txt`

This should contain the general project and notebook dependencies needed to explore the workshop.

### `requirements-train.txt`

This should contain the training stack, such as:

- `transformers`
- `accelerate`
- `peft`
- `trl`

Separating these files helps keep the base setup lighter and reduces platform-specific installation issues.

## PyTorch Note

Some training dependencies may pull in PyTorch automatically. On many systems this works fine, but on Windows ARM or other less common platforms you may need to install a compatible PyTorch build separately or use an x64 Python interpreter.

A common install command is:

```bash
pip install torch torchvision torchaudio
```

If this fails on your platform, consult the official PyTorch installation instructions for a compatible wheel.

## Running the Workshop

Launch Jupyter from the activated environment:

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory.


The emphasis is not just on running training, but on understanding why the model changes behavior and how dataset quality influences the final result.

## Common Issues

### `torch` fails to install

This is usually a platform or wheel compatibility issue. Try:

- using Python 3.11 x64
- installing PyTorch separately
- avoiding ARM-native Python on Windows for training workflows

### `uv` creates the wrong environment

If you are not using the default `.venv` name, activate your environment first and run commands with `--active`:

```bash
uv sync --active
uv run --active jupyter notebook
```

## Repository Structure

```text
model-alignment-lab/
├── configs/                 # Configuration files
├── datasets/                # Training and evaluation datasets
├── models/                  # Saved model artifacts or references
├── notebooks/               # Workshop notebooks
├── outputs/                 # Training outputs and checkpoints
├── scripts/                 # Utility and entry scripts
├── src/                     # Reusable source code
├── venv_x86/                # Local virtual environment (do not commit)
├── .gitignore
├── .python-version
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
├── requirements-train.txt
├── requirements.txt
└── uv.lock
```


## License

See the `LICENSE` file in this repository.
