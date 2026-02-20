"""
Few-Shot Coding Agent (Unified)

This script evaluates LLMs on graph coding tasks using few-shot examples
to guide code generation. The LLM receives example(s) showing the task format
and expected code structure, then generates solutions for new graph problems.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_fs_coding.py --benchmark complexgraph --model deepseek-chat --dataset small --max_instances 10
    python runners/run_fs_coding.py --benchmark complexgraph --model llama --dataset composite --max_instances 10

    # GTools
    python runners/run_fs_coding.py --benchmark gtools --model deepseek-chat --dataset small --max_instances 10
    python runners/run_fs_coding.py --benchmark gtools --model llama --dataset large --max_instances 10

    # With custom number of shots
    python runners/run_fs_coding.py --benchmark complexgraph --model gpt-4 --dataset small --num_shots 3

    # Run specific tasks only
    python runners/run_fs_coding.py --benchmark complexgraph --model gpt-4 --dataset small --tasks clustering diameter

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark
    - Few-shot examples in prompts/fs_code_example_wo_ret.txt (or with composite variant)

Output Files:
    - {task_name}_results.json: Per-task predictions and responses
    - all_results.json: Combined results from all tasks
    - all_results_with_eval.json: Results with evaluation annotations
    - evaluation_metrics.json: Accuracy metrics and statistics
"""

import argparse
import json
import pathlib
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import sys
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables (API keys)
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
    load_test_case_from_file
)
from utils.dataset_config import get_dataset_config


def load_few_shot_examples(num_shots: int = 1, config: Dict[str, Any] = None) -> str:
    """
    Load few-shot code examples for code generation.

    For ComplexGraph composite dataset, uses composite-specific example file.
    For other cases (gtools or non-composite complexgraph), uses standard example.

    Args:
        num_shots: Number of examples (affects which file to load)
        config: Dataset configuration from get_dataset_config()

    Returns:
        String containing the few-shot examples
    """
    # Determine which example file to use
    benchmark = config.get("benchmark", "complexgraph")
    dataset_version = config.get("dataset_version", "small")

    # Use composite example only for complexgraph + composite
    use_composite = (benchmark == "complexgraph" and dataset_version == "composite")

    if use_composite:
        example_file = Path("prompts") / "fs_code_example_composite_wo_ret.txt"
    else:
        example_file = Path("prompts") / "fs_code_example_wo_ret.txt"

    if not example_file.exists():
        print(f"⚠ Warning: Example file not found: {example_file}")
        return ""

    with open(example_file, "r", encoding="utf-8") as f:
        examples = f.read()

    print(f"✓ Loaded few-shot examples from: {example_file}")
    return examples


def create_code_generation_prompt(
    question_text: str,
    is_weighted: bool,
    is_directed: bool,
    args: Optional[Dict[str, Any]],
    few_shot_examples: Optional[str]
) -> str:
    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"

    # Format arguments description
    args_desc = ""
    if args:
        args_list = [f"{k} (variable name: '{k}')" for k in args.keys()]
        args_desc = f"The following variables are provided: {', '.join(args_list)}."

    # Build base prompt
    base_prompt = f"""Given the task description: {question_text}

Generate a Python function that solves this task for a {weighted_text} {directed_text} graph.

Input:
- edge_list: A {weighted_text} {directed_text} graph represented as a list of edges.
  - If weighted: Each edge is [source, target, weight] where weight is a float.
  - If unweighted: Each edge is [source, target].
  - {args_desc}

Output:
- Store your final answer in a variable named 'result'.
- The result should match the expected return type for this task.

Constraints:
- Do NOT use NetworkX or any external graph libraries.
- Implement the algorithm from scratch using only Python standard library.
- Do NOT include test cases or example usage - only the solution function."""

    # Add few-shot examples if provided
    if few_shot_examples:
        prompt = f"""{base_prompt}

EXAMPLE:
{few_shot_examples}

Now, solve the task described above following a similar approach.

Provide your implementation in a Python code block:
"""
    else:
        # Fall back to zero-shot if no examples provided
        prompt = base_prompt + "\n\nProvide your implementation in a Python code block:\n"

    return prompt

def run_code_generation_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None,
    num_shots: int = 1
):
    """
    Run few-shot code generation experiment.

    This function:
    1. Loads the dataset (questions and graphs)
    2. For each question group, creates prompts with few-shot examples
    3. Generates code using the LLM
    4. Executes and evaluates the code
    5. Saves results

    Args:
        model_name: Name of LLM model (e.g., 'gpt-4', 'deepseek-coder')
        dataset_version: Dataset version (e.g., 'small', 'large', 'composite')
        config: Dataset configuration from get_dataset_config()
        output_dir: Custom output directory
        max_instances: Maximum test instances per task
        task_filter: List of specific task names to run
        num_shots: Number of few-shot examples
    """
    print(f"\n{'='*70}")
    print(f"Few-Shot Code Generation Experiment")
    print(f"Model: {model_name}")
    print(f"Dataset: {config['label']}-{dataset_version.upper()}")
    if max_instances:
        print(f"Max instances per task: {max_instances}")
    if task_filter:
        print(f"Task filter: {task_filter}")
    print(f"{'='*70}\n")

    # Initialize LLM generator
    try:
        llm_generator = create_code_generator(model_name)
        print(f"✓ LLM generator initialized: {model_name}\n")
    except Exception as e:
        print(f"✗ Error initializing LLM: {e}")
        return

    # Load dataset
    try:
        questions_data, graphs_data = config["load_data"](dataset_version)
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Set up output directory
    if output_dir is None:
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        output_dir = Path("LLM_generation_results") / config["output_base"] / "code_generation" / "few_shot" / dataset_version / model_short
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load few-shot examples
    config["benchmark"] = config.get("benchmark", "complexgraph")
    config["dataset_version"] = dataset_version
    few_shot_examples = load_few_shot_examples(num_shots=num_shots, config=config)

    # Process each question group
    all_results = []

    for group_idx, question_group in enumerate(questions_data):
        task_name = question_group['task_name']
        is_weighted = question_group['weighted']
        is_directed = question_group['directed']
        graph_instances = question_group['graph_data']

        # Apply task filter
        if task_filter and task_name not in task_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        # Get sample args
        sample_args = graph_instances[0]['args'] if graph_instances else None

        # Create prompt with few-shot examples
        prompt = create_code_generation_prompt(
            question_text=question_text,
            is_weighted=is_weighted,
            is_directed=is_directed,
            args=sample_args,
            few_shot_examples=few_shot_examples
        )

        # Generate code
        try:
            response = llm_generator.generate(prompt)
            code = extract_code_from_response(response)
        except Exception as e:
            print(f"✗ Error generating code: {e}")
            code = None

        # Process instances
        instances_to_process = graph_instances[:max_instances] if max_instances else graph_instances
        task_results = {
            'task_name': task_name,
            'weighted': is_weighted,
            'directed': is_directed,
            'question_text': question_text,
            'generated_code': code,
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
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': 'Graph not found'
                })
                continue

            graph_info = graphs_data[graph_id]
            edge_list = graph_info['graph']

            # Execute code
            try:
                result = execute_code_with_timeout(
                    code=code,
                    edge_list=edge_list,
                    args=args,
                    timeout_seconds=30
                )
                extracted_answer = parse_answer(result, ground_truth)
                is_correct, _ = compare_results_type_aware(extracted_answer, ground_truth)
                if is_correct:
                    correct_count += 1

            except Exception as e:
                error_count += 1
                extracted_answer = None
                is_correct = False

            task_results['predictions'].append({
                'instance_id': inst_idx,
                'graph_id': graph_id,
                'args': args,
                'ground_truth': ground_truth,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct
            })

        # Statistics
        total = len(instances_to_process)
        if total > 0:
            correct_rate = correct_count / total
            task_results['statistics'] = {
                'total_instances': total,
                'correct_count': correct_count,
                'error_count': error_count,
                'correct_rate': correct_rate
            }

            print(f"\n✓ Task complete: {correct_count}/{total} correct ({correct_rate*100:.1f}%)")

        all_results.append(task_results)

        # Save task results
        output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        save_dict_to_json(task_results, str(output_file))

    # Save combined results
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version, 'num_shots': num_shots},
                      str(combined_output))
    print(f"\n✓ Raw results saved to: {combined_output}")

    # Evaluate
    evaluation_results = evaluate_all_results(all_results)
    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}")

    # Combined output
    combined_with_eval = output_dir / "all_results_with_eval.json"
    save_dict_to_json({
        'results': all_results,
        'evaluation': evaluation_results,
        'model': model_name,
        'dataset': dataset_version,
        'num_shots': num_shots
    }, str(combined_with_eval))
    print(f"✓ Results with evaluation saved to: {combined_with_eval}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Few-shot code generation evaluation (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    python runners/run_fs_coding.py --benchmark complexgraph --model gpt-4 --dataset small
    python runners/run_fs_coding.py --benchmark gtools --model deepseek-chat --dataset large
        """
    )

    parser.add_argument("--benchmark", type=str, required=True, choices=["complexgraph", "gtools"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--tasks", type=str, nargs='+', default=None)
    parser.add_argument("--num_shots", type=int, default=1)

    args = parser.parse_args()
    config = get_dataset_config(args.benchmark, args.dataset)

    start_time = time.time()
    run_code_generation_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_instances=args.max_instances,
        task_filter=args.tasks,
        num_shots=args.num_shots
    )
    end_time = time.time()
    print(f"Total elapsed time: {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
