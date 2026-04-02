# Model Alignment Lab

Fine-tuning small language models with LoRA for structured outputs.

## Overview

Model Alignment Lab is a hands-on workshop focused on practical language model alignment. The goal is to take a base language model and adapt its behavior so that it follows instructions more reliably and produces structured, machine-readable outputs such as JSON.

This workshop is designed to help you understand the gap between a general-purpose base model and a model that has been tuned for a specific task format. Rather than retraining a model from scratch, the workshop uses parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation) to shape behavior with far lower compute and memory requirements.

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

You can set up the project with either `uv` or `pip install -e .`.

### Python version

This project currently targets Python 3.11.

On Windows ARM systems, prefer an x64 Python installation if you plan to install PyTorch and related training dependencies.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DerrickJ1612/model-alignment-lab.git
cd model-alignment-lab
```

### Option 1: Setup with uv

This is the preferred workflow for local development.

#### Create the environment and install project dependencies

```bash
uv venv --python 3.11
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
uv sync
```

macOS / Linux:

```bash
source .venv/bin/activate
uv sync
```

If you already have an active custom virtual environment, use:

```bash
uv sync --active
```

### Option 2: Setup with pip

If you are not using `uv`, create a virtual environment first and then install the project in editable mode.

#### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Install the project

```bash
pip install -e .
```

### Option 3: Setup in Google Colab

Colab does not need `uv`. Clone the repository and install the package in editable mode.

```python
!wget https://github.com/<your-username>/model-alignment-lab/archive/refs/heads/main.zip
!unzip main.zip
%cd model-alignment-lab-main
!pip install -e .
```

## PyTorch Note

PyTorch may need to be installed separately depending on your machine and accelerator.

For example, on a Windows desktop with an NVIDIA GPU:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Or with pip:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If you are running CPU-only or using a hosted environment such as Colab, use the PyTorch build appropriate for that system.

## Running the Workshop

Launch Jupyter from the activated environment:

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory.

The emphasis is not just on running training, but on understanding why the model changes behavior and how dataset quality influences the final result.

## Common Issues

### PyTorch installation problems

This is usually a platform or wheel compatibility issue. Try:

- using Python 3.11 x64
- installing PyTorch separately for your platform
- avoiding ARM-native Python on Windows for training workflows

### `uv` uses the wrong environment

If you are not using the default `.venv` name, activate your environment first and run commands with `--active`.

```bash
uv sync --active
uv run --active jupyter notebook
```

## Repository Structure

```text
model-alignment-lab/
├── datasets/
│   └── structured_json/
├── models/
│   └── smollm-travel-router-lora/
├── notebooks/
├── outputs/
├── src/
│   └── model_alignment_lab/
│       ├── evaluation/
│       └── utils/
├── .gitignore
├── .python-version
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
├── requirements-colab.txt
├── requirements.txt
└── uv.lock
```

Note: local virtual environments such as `.venv/` or `.venv_x86/` should not be committed and are intentionally omitted from the project tree above.

## License

See the `LICENSE` file in this repository.
