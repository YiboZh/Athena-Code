# Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning

[![Paper](https://img.shields.io/badge/arXiv-ComingSoon-blue)](https://arxiv.org/abs/2501.XXXX)
[![Conference](https://img.shields.io/badge/NeurIPS-2025--Spotlight-red)](https://nips.cc/virtual/2025/loc/san-diego/poster/118043/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-9cf)](https://yibozh.github.io/Athena/)

---

## 🧩 Overview

Traditional utility-based models explain *what* people would do; **ATHENA** aims to explain *what people actually do*.

![overview](/assets/fig1-final.jpg)

The framework has **two key stages**:

1. **Group-Level Symbolic Utility Discovery** — LLM-guided symbolic regression discovers interpretable, group-level utility functions.
2. **Individual-Level Semantic Adaptation** — personalized templates are optimized with textual gradients to capture individual preferences and constraints.


## Getting Started

**Prerequisites**

- Python 3.10 or newer
- Optional local LLM runtime (e.g. [Ollama](https://ollama.com/)) if you prefer not to use hosted APIs

**Setup**

```bash
git clone https://github.com/your-org/UtilitySymReg.git
cd UtilitySymReg
conda create -n athena python=3.10 -y
conda activate athena
pip install -r requirements.txt
```

**Environment variables** (export the ones that match the backends you plan to call):

| Provider/Backend | Variable | Notes |
| --- | --- | --- |
| OpenAI (`--chatbot openai`) | `OPENAI_API_KEY` | Required for GPT-4o/mini and any TextGrad run using the OpenAI endpoint. |
| Google Gemini (`--chatbot gemini`) | `GEMINI_API_KEY` | Required for Gemini 2.0 Flash. |
| Lambda Labs (`--chatbot lambda`) | `LAMBDA_API_KEY` | Needed for Lambda-hosted DeepSeek models and TextGrad with `base_url=LAMBDA_BASE_URL`. |
| DeepInfra (`--chatbot deepinfra`) | `DEEPINFRA_API_KEY` | Needed for DeepInfra-hosted DeepSeek models and TextGrad with `base_url=DEEPINFRA_BASE_URL`. |

Example setup on macOS/Linux:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export LAMBDA_API_KEY="lam-..."
export DEEPINFRA_API_KEY="din-..."
```

You can store these exports in `~/.zshrc` or your preferred shell profile so they persist across sessions. Local Ollama backends expect a listening server at `http://localhost:11434` (`OllamaAPI`); update the wrapper if your host/port differs.

---

## Running ATHENA Pipelines

ATHENA exposes a CLI via `main.py`. Key parameters include the task (`travel-mode` or `vaccine`), chatbot backend, model identifier, optimisation iterations, and TextGrad steps.

### Travel-mode (SwissMetro)

```bash
python main.py \
  --task travel-mode \
  --chatbot ollama \
  --model deepseek-r1:32b \
  --results-dir results-travel \
  --persona-csv cache/persona_travel.csv \
  --selected-pids-file data/swissmetro/selected_pids_subset_100.txt
```

The command performs group-level symbolic utility discovery, saves intermediate CSVs under `results-travel`, and optimises personalised prompts with TextGrad. Persona strings are cached to `cache/persona_travel.csv` to allow resumable runs.

### Vaccine-choice

```bash
python main.py \
  --task vaccine \
  --chatbot deepinfra \
  --model deepseek-r1:32b \
  --results-dir results-vaccine \
  --selected-pids-file data/vaccine/selected_pids.txt
```

The vaccine pipeline reuses the same CLI knobs except `--persona-csv`, which is not required. By default the run resumes from the latest iteration found in `results-vaccine` and skips recomputation when final CSVs already exist.

### Resuming & Customising

- To pick up from previous progress, leave `--results-dir` pointing to an existing directory; the pipeline auto-detects the last completed iteration.
- Adjust `--top-k`, `--bottom-k`, and `--n-candidates` to control exploration versus exploitation in symbolic search.
- `--tg-steps` governs the number of TextGrad updates per individual.

---

## Persona Cache Utility

For vaccine experiments you can pre-compute personas to avoid repeated API calls:

```bash
python save_persona.py --results-dir results-vaccine --selected-pids-file data/vaccine/selected_pids.txt
```

This populates `results-vaccine/persona.csv`, which `main.py --task vaccine` will reuse.

---

## Baseline Reproduction

The `baseline_model/` directory includes reference implementations for zero-shot, few-shot, and TextGrad-only baselines.

Example (SwissMetro zero-shot):

```bash
python baseline_model/travel_mode/travel_mode_zero_shot.py --chatbot openai --model gpt-4o-mini
```

Example (Vaccine few-shot):

```bash
python baseline_model/vaccine/vaccine_fewshot.py --chatbot gemini --model gemini-2.0-flash
```

Each script shares a similar CLI for selecting chatbot providers and output directories.

---

## Logging & Outputs

- Logs follow the configuration in `athena/config/logging.json` and are written to `athena/config/logs/` by default.
- Group-level optimisation artefacts are stored as CSV files named `utility_functions_results_group_{gid}_iteration_{k}.csv` inside the chosen `results_dir`.
- Persona caches live inside `results_dir/persona.csv`.

---

## Citation

```bibtex
@inproceedings{zhao2025athena,
  title        = {Personalized Decision Modeling: Utility Optimization or Textualized-Symbolic Reasoning},
  author       = {Yibo Zhao, Yang Zhao, Hongru Du, Hao Frank Yang},
  booktitle    = {The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year         = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).


