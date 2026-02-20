# Data Pipeline

This directory contains scripts for data generation, labeling, and preprocessing.

## Structure

### dataset_generation/
- `graph_dataset_gen.py` - Core dataset generation utilities
  - Functions: `save_dict_to_json()`, graph generation helpers
- `gwild_determine_weightedness_directedness.py` - Analyze graph properties for GWild dataset

### labeling/
- `add_labels_graphtutor.py` - Add labels to GraphTutor dataset
  - Supports different graph sizes via command-line arguments

### args_generation/
- `add_extra_args_graphtutor.py` - Generate additional arguments for GraphTutor tasks
  - Configurable for different dataset variants

## Usage Examples

### Generate labeled data:
```bash
python data_pipeline/labeling/add_labels_graphtutor.py --size small
```

### Add extra arguments:
```bash
python data_pipeline/args_generation/add_extra_args_graphtutor.py --size large
```

### Analyze graph properties:
```bash
python data_pipeline/dataset_generation/gwild_determine_weightedness_directedness.py
```
