# Repository Guidelines

## Project Structure & Module Organization
This repository focuses on backdoor detection for MoE-based LLMs. Key directories:
- `classification_backdoors/attacks/`: attack and finetuning scripts (e.g., `step0_finetune.py`, `step1_poisoned_data_preparation.py`).
- `classification_backdoors/detection/`: routing-recording and analysis scripts.
- `evaluation/`: evaluation scripts and launchers for routing analysis.
- `configs/`: DeepSpeed, FSDP, and accelerator configs (`.json`, `.yaml`).
- `data/`: datasets and generated poisoned/triggered JSONL files.
- `runs/` and `output/`: training checkpoints and evaluation artifacts.
- `diagnostics/`: NCCL and distributed sanity checks.

## Build, Test, and Development Commands
This is a Python/DeepSpeed workflow; typical entry points are shell scripts or direct Python.
- Install deps: `pip install -r requirement_dp.txt`
- Download model weights: `bash download.sh` (uses `huggingface-cli`).
- Finetune attack model: `bash classification_backdoors/attacks/step0_finetune.sh`
- Generate poisoned data: `bash classification_backdoors/attacks/step1.sh`
- Record routing stats: `bash classification_backdoors/detection/step0_record.sh`
- Run routing eval: `bash evaluation/routing_eval.sh`

## Coding Style & Naming Conventions
- Python scripts are CLI-driven with `snake_case` flags; keep new args consistent.
- Configs live in `configs/` and should be referenced by absolute or repo-relative paths.
- Keep outputs in `runs/`, `output/`, or `assets/` to avoid polluting the repo root.
- No formatter is configured; follow the existing script style and keep changes minimal.

## Testing Guidelines
- No unit-test framework is defined. Validation is done via evaluation scripts in `evaluation/`.
- Name new evaluation outputs with timestamps or dataset names (see `output/*.json`).
- If adding tests, document how to run them in this file.

## Commit & Pull Request Guidelines
- Recent commits use short date-stamps or brief messages (e.g., `20251202`, `Add files via upload`).
- Prefer concise, imperative commit messages: `Add routing metrics export`.
- PRs should describe the dataset/model used, link related issues, and include sample outputs
  (paths under `output/` or `runs/`) when changing evaluation or training logic.

## Security & Configuration Tips
- Hugging Face and W&B may require credentials; export tokens in your shell (do not commit).
- Check CUDA/NCCL env vars in the scripts before launching multi-GPU runs.
