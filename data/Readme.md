# Data Guide

This file describes dataset locations and JSON structure used by GraphSkill.

## What Lives In `data/`

- `networkx_graph_functions_docs.json`: retrieval corpus used by TF-IDF / Sentence-BERT / retrieval-agent baselines.
- `retrieval_groundtruth.json`: retrieval evaluation references.
- `retrieval_groundtruth_generation.py`: helper script for creating retrieval ground truth.

## External Benchmark Datasets

Benchmark loaders in `utils/complexgraph_utils.py` and `utils/gtools_utils.py` use:

- ComplexGraph root: `data/ComplexGraph`
- GTools root: `data/GTools`

### ComplexGraph directory layout

Expected subdirectories:

- `data/ComplexGraph/small`
- `data/ComplexGraph/large`
- `data/ComplexGraph/composite`

Each split contains:

- `questions.json`
- `graphs.json`

### GTools directory layout

Expected subdirectories:

- `data/GTools/small`
- `data/GTools/large`

Each split contains:

- `questions.json`
- `graphs.json`

## JSON Schema (Practical Summary)

### `questions.json` (both benchmarks)

Top-level type: `List[question_group]`

Common fields in each `question_group`:

- `task_name`
- `question`
- `weighted`
- `directed`
- `graph_data`: list of test instances

Each `graph_data` item:

- `graph`: graph id (string key into `graphs.json`)
- `args`: task arguments (object, may be empty)
- `answer`: ground-truth answer

ComplexGraph usually includes extra prompt variants in each group:

- `question_no_term`
- `question_paraphrase`
- `question_real_world_1`
- `question_real_world_2`

### `graphs.json`

Top-level type: `Dict[str, graph_record]`

ComplexGraph `graph_record` fields:

- `graph`: edge list
- `weighted`
- `directed`
- `generation_type`
- `connected`
- `cyclic`

GTools `graph_record` fields:

- `graph`: edge list
- `metadata`: conversion/source metadata

## Download Notes

- ComplexGraph data link: [Google Drive](https://drive.google.com/file/d/1hK8soY4QFzSnb2TkufobbDT-rFE6mpX1/view?usp=drive_link)
- After downloading, ensure files are placed under the exact paths above (or update base-path constants in `utils/complexgraph_utils.py` / `utils/gtools_utils.py`).