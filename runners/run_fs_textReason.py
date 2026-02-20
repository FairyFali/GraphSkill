"""
Few-Shot Textual Reasoning Baseline (Unified)

This script evaluates LLMs on graph reasoning tasks using few-shot textual reasoning
without code generation. The LLM receives example(s) showing the task format,
followed by a new graph problem to solve.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_fs_textReason.py --benchmark complexgraph --model deepseek-chat --dataset small --max_instances 10
    python runners/run_fs_textReason.py --benchmark complexgraph --model llama --dataset composite --max_instances 10

    # GTools
    python runners/run_fs_textReason.py --benchmark gtools --model deepseek-chat --dataset small --max_instances 10
    python runners/run_fs_textReason.py --benchmark gtools --model llama --dataset large --max_instances 10

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark
    - Few-shot examples in prompts/fs_text_example.txt

Output Files:
    - {task_name}_results.json: Per-task predictions and responses
    - all_results.json: Combined results from all tasks
    - evaluation_metrics.json: Accuracy metrics and statistics
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

# Load environment variables (API keys)
try:
    import load_env
    print("✓ API keys loaded from .env file")
except ImportError:
    print("⚠ Warning: load_env not found, ensure API keys are set manually")

from utils.get_llm_response_generator import create_code_generator
from utils.shared.json_utils import save_dict_to_json
from utils.complexgraph_utils import *
from utils.dataset_config import get_dataset_config


def load_few_shot_examples(num_shots: int = 1) -> str:
    """
    Load universal few-shot textual reasoning example.

    Args:
        num_shots: Number of examples (default: 1)

    Returns:
        String containing the few-shot example
    """
    examples_path = Path("prompts") / "fs_text_example.txt"

    if not examples_path.exists():
        raise FileNotFoundError(f"Few-shot example file not found: {examples_path}")

    with open(examples_path, "r", encoding="utf-8") as f:
        few_shot_content = f.read()

    print(f"✓ Loaded few-shot textual reasoning example from: {examples_path}")
    return few_shot_content


def create_reasoning_prompt(
    question_text: str,
    edge_list: List[List],
    args: Optional[Dict[str, Any]],
    is_weighted: bool,
    is_directed: bool,
    few_shot_examples: Optional[str] = None
) -> str:
    """Create a prompt for few-shot textual reasoning."""
    graph_structure = format_graph_for_prompt(edge_list, is_weighted, is_directed)
    args_text = format_args_for_prompt(args)

    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"

    prompt = f"""You are an expert in graph theory and algorithms. Your task is to solve the following graph problem using textual reasoning.

TASK DESCRIPTION:
{question_text}

GRAPH PROPERTIES:
- Type: {directed_text}, {weighted_text}
- Number of edges: {len(edge_list)}

EXAMPLES:
{few_shot_examples if few_shot_examples else ""}
{"Now, solve the following problem:" if few_shot_examples else ""}
{"---" if few_shot_examples else ""}

GRAPH STRUCTURE (Edge List):
{graph_structure}
{args_text}

INSTRUCTIONS:
1. Carefully analyze the graph structure provided above
2. Apply graph theory principles to solve the problem
3. State your final answer explicitly (e.g., "Final Answer: 42" or "Final Answer: [1, 2, 3]")

Your response:
"""
    return prompt


def run_textual_reasoning_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None,
    num_shots: int = 1
):
    """
    Run few-shot textual reasoning experiment.

    Args:
        model_name: Name of LLM model
        dataset_version: Dataset version
        config: Dataset configuration from get_dataset_config()
        output_dir: Custom output directory
        max_instances: Maximum test instances per task
        task_filter: List of specific task names to run
        num_shots: Number of few-shot examples
    """
    print(f"\n{'='*70}")
    print(f"Few-Shot Textual Reasoning Experiment")
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
        output_dir = Path("LLM_generation_results") / config["output_base"] / "textual_reasoning" / "few_shot" / dataset_version / model_short
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load few-shot examples
    few_shot_examples_str = load_few_shot_examples(num_shots=num_shots)

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
        print(f"Question Group {group_idx + 1}/{len(questions_data)}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        # Limit instances if specified
        instances_to_process = graph_instances[:max_instances] if max_instances else graph_instances

        # Results for this task
        task_results = {
            'task_name': task_name,
            'weighted': is_weighted,
            'directed': is_directed,
            'question_text': question_text,
            'predictions': []
        }

        # Process each graph instance
        for inst_idx, instance in enumerate(tqdm(instances_to_process, desc=f"Processing {task_name}")):
            graph_id = instance['graph']
            args = instance['args']
            ground_truth = instance['answer']

            # Get graph structure
            if graph_id not in graphs_data:
                continue

            graph_info = graphs_data[graph_id]
            edge_list = graph_info['graph']

            # Create prompt
            prompt = create_reasoning_prompt(
                question_text=question_text,
                edge_list=edge_list,
                args=args,
                is_weighted=is_weighted,
                is_directed=is_directed,
                few_shot_examples=few_shot_examples_str
            )

            # Generate response
            try:
                response = llm_generator.generate(prompt)
                extracted_answer = extract_answer_from_response(response)
                extracted_answer = str(parse_answer(extracted_answer, ground_truth))
                extracted_answer = clean_extracted_answer(extracted_answer)
                extracted_answer = parse_answer_text(extracted_answer, ground_truth)

                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'llm_response': response,
                    'extracted_answer': extracted_answer
                })

            except Exception as e:
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': str(e)
                })

        # Calculate statistics for this task
        total_instances = len(instances_to_process)
        correct_count = 0
        error_count = 0

        for pred in task_results['predictions']:
            if 'error' in pred:
                error_count += 1
            elif 'extracted_answer' in pred:
                is_correct, _ = compare_results_type_aware(pred['extracted_answer'], pred['ground_truth'])
                if is_correct:
                    correct_count += 1

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

        # Save intermediate results
        output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        save_dict_to_json(task_results, str(output_file))

    # Save combined results
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version, 'num_shots': num_shots},
                      str(combined_output))
    print(f"\n✓ Raw results saved to: {combined_output}")

    # Evaluate results
    evaluation_results = evaluate_all_results(all_results)

    # Save evaluation metrics
    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Few-shot textual reasoning evaluation (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    python runners/run_fs_textReason.py --benchmark complexgraph --model gpt-4 --dataset small
    python runners/run_fs_textReason.py --benchmark gtools --model deepseek-chat --dataset large
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
    run_textual_reasoning_experiment(
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
