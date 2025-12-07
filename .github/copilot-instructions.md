**Purpose**
- **What:** Short, practical instructions for AI coding agents to be immediately productive in this repository.
- **Where:** This file lives at the repo root under `.github/copilot-instructions.md`.

**Big Picture**
- **Project type:** Python research / reproducibility repo for an NLP final project (training, evaluation, and analysis). Key areas: data (`datasets/`), training & models (`scripts/`, `cartography_model/`, `models/`), analysis (`analysis_scripts/`, `results/`, `deliverables/`), and notebooks (`notebooks/`).
- **Data flow:** raw JSONL datasets in `datasets/` -> training scripts in `scripts/` or Colab notebooks -> model artifacts in `models/` or `cartography_model/` -> evaluation outputs in `results/` -> analysis scripts in `analysis_scripts/` and visualizations in `deliverables/`.

**Where to look first (high-value files)**
- `requirements.txt` — install runtime deps.
- `nlp-fp/` — optional virtualenv used by authors; contains `Scripts/Activate.ps1` (PowerShell activation example).
- `scripts/train_with_cartography.py` — primary local training entrypoint.
- `scripts/run.py` — convenience runner for common tasks (inspect for supported commands).
- `scripts/create_visualizations.py` and `scripts/create_acm_document.py` — post-processing and deliverable generation.
- `analysis_scripts/` — contains all reproducible analyses; follow naming conventions (e.g., `basic_prediction_analysis.py`, `dataset_cartography.py`).
- `cartography_model/` and `models/` — saved model artifacts and tokenizers (do not modify large binary checkpoints in PRs).
- `colab_assist/` — Colab-first helpers and alternative training flows used in notebooks.

**Environment & common commands**
- Activate local venv (PowerShell):
  - `& "nlp-fp\Scripts\Activate.ps1"`
- Install deps: `pip install -r requirements.txt`
- Run a script: `python scripts/run.py` or `python scripts/train_with_cartography.py`
- Run analyses: `python analysis_scripts/basic_prediction_analysis.py` (inspect script args first)
- Run tests: run the tests folder with your test runner, e.g. `pytest tests/` or `python -m pytest tests/`.

**Repository conventions & patterns**
- Single-purpose scripts live in `scripts/` and are callable from the repo root using `python`.
- Analysis code in `analysis_scripts/` expects to read model outputs from `results/` and `cartography_model/` — prefer to generate outputs in `results/` so downstream analyses work.
- Models and tokenizers are stored alongside training metadata under `cartography_model/` (contains `model.safetensors`, `tokenizer.json`, `train_results.json`, `eval_results.json`). Treat these directories as data/artifacts rather than source code.
- Notebooks under `notebooks/` are Colab-ready and can reference `colab_assist/` utilities; they are for exploratory/interactive use, not unit-tests.

**Merging & PR guidance for an AI assistant**
- Do not change large model artifacts or `vocab/tokenizer` files.
- For code changes, run local scripts and at least one analysis script to ensure outputs are compatible (e.g., run `scripts/create_visualizations.py` or `analysis_scripts/basic_prediction_analysis.py`).
- If adding or changing datasets, follow existing JSONL format in `datasets/` and update any consumers in `scripts/` or `analysis_scripts/`.

**Examples of project-specific edits**
- To add a CLI flag consumed by analyses: update the script in `scripts/`, then update calls inside `analysis_scripts/` that expect the produced filenames.
- To add a new visualization: add code to `scripts/create_visualizations.py` and add the generated artifact path to `deliverables/visualizations/`.

**Integration points & external deps**
- Colab integration: `colab_assist/` scripts and `notebooks/NLP_Final_Project_Colab.ipynb` (use the Colab helpers when replicating remote training flows).
- Training uses typical Hugging Face-style artifacts (see `cartography_model/` for file names). External storage or large files may be committed; assume large files are not edited in PRs.

**If something is unclear**
- Run quick discovery commands: `python -m pip install -r requirements.txt` then `python scripts/run.py --help` and inspect `analysis_scripts/` headers for expected inputs.

**Contact / next steps**
- After making edits, run one short analysis or visualization to validate integration and include the command used in the PR description.

Please review and tell me which areas you'd like expanded (examples, more command snippets, or linking exact script args).
