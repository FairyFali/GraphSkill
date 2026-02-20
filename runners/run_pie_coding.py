"""
PIE (Pseudocode-Injection Example) + Coding Agent (Unified)

This script implements code generation with pre-retrieved pseudocode examples
to guide the LLM in generating correct implementations.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_pie_coding.py --benchmark complexgraph --model deepseek-chat --dataset small
    python runners/run_pie_coding.py --benchmark complexgraph --model llama --dataset composite

    # GTools
    python runners/run_pie_coding.py --benchmark gtools --model deepseek-chat --dataset small
    python runners/run_pie_coding.py --benchmark gtools --model llama --dataset large

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark
    - Pre-retrieved pseudocode examples in prompts/graph_tasks_pseudocode/
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import sys
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    import load_env
    print("✓ API keys loaded from .env file")
except ImportError:
    print("⚠ Warning: load_env not found, ensure API keys are set manually")

from utils.get_llm_response_generator import create_code_generator
from utils.shared.json_utils import save_dict_to_json
from utils.complexgraph_utils import *
from utils.code_execution_utils import (
    extract_code_from_response,
    execute_code_with_timeout,
    compare_results_type_aware
)
from utils.complexgraph_codingagent_utils import (
    create_code_generation_prompt,
    load_test_case_from_file
)
from utils.dataset_config import get_dataset_config


def load_pseudocode_examples(
    task_name: str,
    is_directed: bool,
    is_weighted: bool,
    pseudo_examples_dir: Path = Path("prompts/graph_tasks_pseudocode")
) -> List[str]:
    """
    Load pre-retrieved pseudocode examples for a specific task.

    Args:
        task_name: Name of the graph algorithm task
        is_directed: Whether the graph is directed
        is_weighted: Whether the graph is weighted
        pseudo_examples_dir: Base directory containing pseudocode examples

    Returns:
        List of pseudocode/example strings for this task
    """
    try:
        direction_str = "directed" if is_directed else "undirected"
        weight_str = "weighted" if is_weighted else "unweighted"

        pseudo_file = pseudo_examples_dir / direction_str / weight_str / "pseudo_scripts.json"

        if not pseudo_file.exists():
            return []

        with open(pseudo_file, 'r', encoding='utf-8') as f:
            pseudo_data = json.load(f)

        if task_name not in pseudo_data:
            return []

        task_pseudo = pseudo_data[task_name]
        if isinstance(task_pseudo, str):
            return [task_pseudo]
        elif isinstance(task_pseudo, list):
            return task_pseudo
        else:
            return [str(task_pseudo)]

    except Exception as e:
        return []


def run_pie_coding_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    pseudo_examples_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_correction_rounds: int = 1,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None
):
    """
    Run PIE (Pseudocode-injection) + coding agent experiment.

    Args:
        model_name: Name of LLM model
        dataset_version: Dataset version
        config: Dataset configuration from get_dataset_config()
        pseudo_examples_dir: Directory containing pre-retrieved pseudocode examples
        output_dir: Custom output directory
        max_correction_rounds: Maximum error correction iterations
        max_instances: Maximum test instances per task
        task_filter: List of specific task names to run
    """
    print(f"\n{'='*70}")
    print(f"PIE (Pseudocode-injection) + Coding Agent Experiment")
    print(f"Model: {model_name}")
    print(f"Dataset: {config['label']}-{dataset_version.upper()}")
    print(f"Max correction rounds: {max_correction_rounds}")
    print(f"{'='*70}\n")

    # Initialize LLM
    try:
        llm_generator = create_code_generator(model_name)
        print(f"✓ LLM generator initialized: {model_name}\n")
    except Exception as e:
        print(f"✗ Error initializing LLM: {e}")
        return

    # Load dataset
    try:
        questions_data, graphs_data = config["load_data"](dataset_version)
        print(f"✓ Loaded {config['label']} dataset\n")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Set up output directory
    if output_dir is None:
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        output_dir = Path("LLM_generation_results") / config["output_base"] / "code_generation" / "pie_coding_agent" / dataset_version / model_short
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set pseudocode directory
    if pseudo_examples_dir is None:
        pseudo_examples_dir = Path("prompts/graph_tasks_pseudocode")

    all_results = []

    # Process each question group
    for group_idx, question_group in enumerate(questions_data):
        task_name = question_group['task_name']
        is_weighted = question_group['weighted']
        is_directed = question_group['directed']
        graph_instances = question_group['graph_data']

        if task_filter and task_name not in task_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        # Load pre-retrieved pseudocode examples
        pseudocode_examples = load_pseudocode_examples(
            task_name=task_name,
            is_directed=is_directed,
            is_weighted=is_weighted,
            pseudo_examples_dir=pseudo_examples_dir
        )

        print(f"✓ Loaded {len(pseudocode_examples)} pseudocode example(s)")

        # Load test case
        test_case = load_test_case_from_file(
            test_case_path=config["test_case_path"],
            task_name=task_name,
            is_directed=is_directed,
            is_weighted=is_weighted
        )

        sample_args = graph_instances[0]['args'] if graph_instances else None

        # Create prompt with pseudocode examples
        prompt = create_code_generation_prompt(
            question_text=question_text,
            is_weighted=is_weighted,
            is_directed=is_directed,
            args=sample_args,
            retrieved_docs=pseudocode_examples,
            retrieval_method="Pseudocode injection",
            test_case=test_case
        )

        # Generate code
        code = None
        try:
            response = llm_generator.generate(prompt)
            code = extract_code_from_response(response)
        except Exception as e:
            print(f"✗ Error generating code: {e}")

        # Process instances
        instances_to_process = graph_instances[:max_instances] if max_instances else graph_instances
        task_results = {
            'task_name': task_name,
            'weighted': is_weighted,
            'directed': is_directed,
            'generated_code': code,
            'pseudocode_examples': pseudocode_examples,
            'num_examples': len(pseudocode_examples),
            'predictions': []
        }

        correct_count = 0
        error_count = 0

        for inst_idx, instance in enumerate(tqdm(instances_to_process, desc=f"Processing {task_name}")):
            graph_id = instance['graph']
            args = instance['args']
            ground_truth = instance['answer']

            if graph_id not in graphs_data:
                error_count += 1
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'ground_truth': ground_truth,
                    'error': 'Graph not found',
                    'is_correct': False
                })
                continue

            edge_list = graphs_data[graph_id]['graph']

            # Execute code with timeout protection
            success = False
            result = None
            error_msg = None
            try:
                result = execute_code_with_timeout(
                    code=code,
                    edge_list=edge_list,
                    args=args,
                    timeout_seconds=30
                )
                success = True
            except Exception as e:
                error_msg = str(e)
                success = False

            # Parse result
            prediction = parse_answer(result, ground_truth) if success else None

            # Compare with ground truth
            is_correct = False
            if prediction is not None and ground_truth is not None:
                is_correct, _ = compare_results_type_aware(prediction, ground_truth)

            # Update statistics
            if not success:
                error_count += 1
            elif is_correct:
                correct_count += 1

            # Record result
            task_results['predictions'].append({
                'instance_id': inst_idx,
                'graph_id': graph_id,
                'ground_truth': ground_truth,
                'extracted_answer': prediction,
                'is_correct': is_correct,
                'execution_error': error_msg if not success else None
            })

        # Calculate statistics
        total_instances = len(instances_to_process)

        if total_instances > 0:
            correct_rate = correct_count / total_instances
            error_rate = error_count / total_instances

            task_results['statistics'] = {
                'total_instances': total_instances,
                'correct_count': correct_count,
                'error_count': error_count,
                'correct_rate': correct_rate,
                'error_rate': error_rate
            }

            print(f"\n✓ Task complete: {correct_count}/{total_instances} correct ({correct_rate*100:.1f}%)")

        all_results.append(task_results)

        # Save task results
        task_output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        save_dict_to_json(task_results, str(task_output_file))

    # Save combined results
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version},
                      str(combined_output))
    print(f"\n✓ Raw results saved to: {combined_output}")

    # Evaluate all results
    evaluation_results = evaluate_all_results(all_results)
    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PIE (Pseudocode-injection) + Coding Agent (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--benchmark", type=str, required=True, choices=["complexgraph", "gtools"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="small")
    parser.add_argument("--pseudo_dir", type=str, default="prompts/graph_tasks_pseudocode")
    parser.add_argument("--max_correction_rounds", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--tasks", type=str, nargs='+', default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    config = get_dataset_config(args.benchmark, args.dataset)

    start_time = time.time()
    run_pie_coding_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        config=config,
        pseudo_examples_dir=Path(args.pseudo_dir) if args.pseudo_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_correction_rounds=args.max_correction_rounds,
        max_instances=args.max_instances,
        task_filter=args.tasks
    )
    end_time = time.time()
    print(f"Total elapsed time: {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
