# GraphSkill

GraphSkill is a Python research framework for evaluating LLMs on graph reasoning tasks through both code generation and text reasoning pipelines.

It currently supports two benchmarks:
- `complexgraph`
- `gtools`

The repository includes multiple runner scripts for zero-shot, few-shot, chain-of-thought, retrieval-assisted, and agentic baselines, with standardized result outputs and evaluation metrics.

## Repository Structure

- `runners/` - experiment entry points (`run_*.py`)
- `utils/` - shared utilities (dataset loading, LLM wrappers, evaluation, code execution)
- `prompts/` - benchmark test cases and prompting assets
- `data/` - retrieval/documentation data and supporting JSON resources
- `LLM_generation_results/` - generated outputs and evaluation artifacts
- `.env.example` - environment variable template for API keys

## Prerequisites

- Python 3
- API keys for whichever model providers you plan to use (for example OpenAI, DeepSeek, Together, Hugging Face)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development tooling (tests, linting, formatting):

```bash
pip install -r requirements-dev.txt
```

## Environment Setup

Create a local `.env` file and populate it with your API keys:

```bash
cp .env.example .env
```

Then load variables into your shell:

```bash
source load_env.sh
```

## Quick Start

Run zero-shot code generation baseline:

```bash
python runners/run_zs_coding.py --benchmark complexgraph --model deepseek-chat --dataset small --max_instances 10
```

Run CodeGraph baseline (code generation with retrieved examples):

```bash
python runners/run_codegraph.py --benchmark gtools --model deepseek-chat --dataset small --max_instances 10
```

Evaluate an existing results file without re-running generation:

```bash
python runners/run_zs_coding.py --benchmark complexgraph --evaluate_only --results_file ./LLM_generation_results/complexgraph/code_generation/zero_shot/small/<model>/all_results.json
```

## Benchmark and Dataset Options

- `--benchmark`: `complexgraph` or `gtools`
- `--dataset`:
  - `complexgraph`: `small`, `large`, `composite`
  - `gtools`: `small`, `large`

Use each runner's built-in help for full arguments:

```bash
python runners/run_zs_coding.py --help
python runners/run_codegraph.py --help
```

## Output Artifacts

Runners save outputs under `LLM_generation_results/...`, typically including:
- per-task `*_results.json`
- `all_results.json`
- `all_results_with_eval.json`
- `evaluation_metrics.json`

## Available Runner Scripts

The repo contains several baselines under `runners/`, including:
- `run_zs_coding.py`
- `run_codegraph.py`
- `run_fs_coding.py`
- `run_pie_coding.py`
- `run_graphteam_coding.py`
- `run_zs_textReason.py`
- `run_fs_textReason.py`
- `run_cot_textReason.py`
- `run_bag_textReason.py`
- `run_zs_retrieval_textReason.py`
- `run_zs_sentbert_codingagent.py`
- `run_zs_tfidf_codingagent.py`
- `run_zs_retagent_codingagent.py`

## Notes

- Keep `.env` local and never commit secrets.
- Large datasets can require longer runtime due to code execution and evaluation.

## 📃 Citation
```
@misc{wang2026graphskill,
      title={GraphSkill: Documentation-Guided Hierarchical Retrieval-Augmented Coding for Complex Graph Reasoning}, 
      author={Fali Wang and Chenglin Weng and Xianren Zhang and Siyuan Hong and Hui Liu and Suhang Wang},
      year={2026},
      eprint={2603.06620},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2603.06620}, 
}
```

## License

This project is distributed under the terms in `LICENSE`.
