"""
Zero-Shot Textual Reasoning Baseline (Unified)

This script evaluates LLMs on graph reasoning tasks using direct textual reasoning
without code generation. The LLM receives a natural language task description and
graph structure, then provides an answer based on graph theory reasoning.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_zs_textReason.py --benchmark complexgraph --model deepseek-chat --dataset small --max_instances 10
    python runners/run_zs_textReason.py --benchmark complexgraph --model llama --dataset composite --max_instances 10

    # GTools
    python runners/run_zs_textReason.py --benchmark gtools --model deepseek-chat --dataset small --max_instances 10
    python runners/run_zs_textReason.py --benchmark gtools --model llama --dataset large --max_instances 10

    # Evaluate existing results
    python runners/run_zs_textReason.py --benchmark complexgraph --evaluate_only --results_file ./results/all_results.json

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark

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
from utils.dataset_config import get_dataset_config



def create_reasoning_prompt(question_text: str, edge_list: List[List],
                           args: Optional[Dict[str, Any]],
                           is_weighted: bool, is_directed: bool) -> str:
    """
    Create a comprehensive prompt for zero-shot textual reasoning.

    The prompt includes:
    1. Task description from question text
    2. Graph structure in natural language
    3. Task parameters (if any)
    4. Instructions for reasoning without code

    Args:
        question_text: Natural language task description
        edge_list: Graph structure as edge list
        args: Task-specific arguments (node, source/target, etc.)
        is_weighted: Whether graph is weighted
        is_directed: Whether graph is directed

    Returns:
        Complete prompt for the LLM
    """
    # Format graph structure
    graph_structure = format_graph_for_prompt(edge_list, is_weighted, is_directed)

    # Format arguments if present
    args_text = format_args_for_prompt(args)

    # Determine graph properties text
    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"

    # Build the prompt
    prompt = f"""You are an expert in graph theory and algorithms. Your task is to solve the following graph problem using textual reasoning.

        TASK DESCRIPTION:
        {question_text}

        GRAPH PROPERTIES:
        - Type: {directed_text}, {weighted_text}
        - Number of edges: {len(edge_list)}

        GRAPH STRUCTURE (Edge List):
        {graph_structure}
        {args_text}

        INSTRUCTIONS:
        1. Carefully analyze the graph structure provided above
        2. Apply graph theory principles to solve the problem
        3. State your final answer explicitly (e.g., "Final Answer: 42" or "Final Answer: [1, 2, 3]")

        IMPORTANT CONSTRAINTS:
        - Do NOT write any code or pseudocode
        - Base your answer purely on graph theory reasoning

        Your response: \n
        """

        # INSTRUCTIONS:
        # 1. Carefully analyze the graph structure provided above
        # 2. Apply graph theory principles to solve the problem
        # 3. Think step-by-step through your reasoning process
        # 4. State your final answer explicitly (e.g., "Final Answer: 42" or "Final Answer: [1, 2, 3]")

    return prompt


def run_textual_reasoning_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None
):
    """
    Run zero-shot textual reasoning experiment.

    This function:
    1. Loads the dataset (questions and graphs)
    2. For each question group, generates reasoning prompts
    3. Queries the LLM for answers
    4. Saves results with predictions and ground truth

    Args:
        model_name: Name of LLM model (e.g., 'gpt-4', 'deepseek-coder')
        dataset_version: Dataset version (e.g., 'small', 'large', 'composite')
        config: Dataset configuration from get_dataset_config()
        output_dir: Custom output directory
        max_instances: Maximum test instances per task (default: all)
        task_filter: List of specific task names to run (default: all)
    """
    print(f"\n{'='*70}")
    print(f"Zero-Shot Textual Reasoning Experiment")
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
        output_dir = Path("LLM_generation_results") / config["output_base"] / "textual_reasoning" / "zero_shot" / dataset_version / model_short
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each question group
    all_results = []

    for group_idx, question_group in enumerate(questions_data):
        task_name = question_group['task_name']
        is_weighted = question_group['weighted']
        is_directed = question_group['directed']
        graph_instances = question_group['graph_data']

        # Apply task filter if specified
        if task_filter and task_name not in task_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Question Group {group_idx + 1}/{len(questions_data)}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"{'='*60}\n")

        # Use the base question text (can also use variants like question_real_world_1)
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
                print(f"⚠ Warning: Graph {graph_id} not found in graphs.json")
                continue

            graph_info = graphs_data[graph_id]
            edge_list = graph_info['graph']

            # Create prompt
            prompt = create_reasoning_prompt(
                question_text=question_text,
                edge_list=edge_list,
                args=args,
                is_weighted=is_weighted,
                is_directed=is_directed
            )

            # Generate response
            try:

                response = llm_generator.generate(prompt)
                # print("DEBUG: response = ", response)
                extracted_answer = extract_answer_from_response(response)
                # print("### DEBUG: extracted_answer before cleaning =", extracted_answer, type(extracted_answer))
                extracted_answer = str(parse_answer(extracted_answer, ground_truth))
                extracted_answer = clean_extracted_answer(extracted_answer)
                # print("### DEBUG: extracted_answer after cleaning =", extracted_answer, type(extracted_answer))
                extracted_answer = parse_answer_text(extracted_answer, ground_truth)
                # print("### DEBUG: extracted_answer after parsing =", extracted_answer, type(extracted_answer))
                # print("### DEBUG: ground_truth =", ground_truth, type(ground_truth))

                # Store result
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'llm_response': response,
                    'extracted_answer': extracted_answer,
                    'prompt_length': len(prompt)
                })

            except Exception as e:
                print(f"\n✗ Error processing instance {inst_idx}: {e}")
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': str(e)
                })

        # Calculate and display statistics for this task
        total_instances = len(instances_to_process)
        correct_count = 0
        incorrect_count = 0
        error_count = 0
        for pred in task_results['predictions']:
            if 'error' in pred:
                error_count += 1
            elif 'extracted_answer' in pred:
                is_correct, _ = compare_results_type_aware(pred['extracted_answer'], pred['ground_truth'])
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

        if total_instances > 0:
            correct_rate = correct_count / total_instances
            error_rate = error_count / total_instances
            incorrect_rate = incorrect_count / total_instances

            print(f"\n{'='*60}")
            print(f"Task Statistics for {task_name}:")
            print(f"  Total instances: {total_instances}")
            print(f"  Correct: {correct_count} ({correct_rate*100:.1f}%)")
            print(f"  Incorrect: {incorrect_count} ({incorrect_rate*100:.1f}%)")
            print(f"  Errors: {error_count} ({error_rate*100:.1f}%)")
            print(f"{'='*60}\n")

            # Add statistics to task results
            task_results['statistics'] = {
                'total_instances': total_instances,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'error_count': error_count,
                'correct_rate': correct_rate,
                'incorrect_rate': incorrect_rate,
                'error_rate': error_rate
            }

        # Add task results to overall results
        all_results.append(task_results)

        # Save intermediate results after each task
        output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        save_dict_to_json(task_results, str(output_file))
        print(f"✓ Saved results to: {output_file}")

    # Save combined results (before evaluation)
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version},
                      str(combined_output))
    print(f"✓ Raw results saved to: {combined_output}")

    # print("DEBUG: all_results =", all_results)

    # Evaluate results
    evaluation_results = evaluate_all_results(all_results)

    # Save evaluation metrics
    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}")

    # Save results with evaluation annotations
    combined_with_eval = output_dir / "all_results_with_eval.json"
    save_dict_to_json({
        'results': all_results,
        'evaluation': evaluation_results,
        'model': model_name,
        'dataset': dataset_version
    }, str(combined_with_eval))
    print(f"✓ Results with evaluation saved to: {combined_with_eval}")

    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"Results directory: {output_dir}")
    print(f"Question groups processed: {len(all_results)}")
    print(f"Overall accuracy: {evaluation_results['overall_metrics']['overall_accuracy']:.2f}%")
    print(f"{'='*70}\n")

    return all_results, evaluation_results


def main():
    """
    Main entry point with argument parsing.

    Parses command-line arguments and runs the textual reasoning experiment.
    """
    parser = argparse.ArgumentParser(
        description="Zero-shot textual reasoning evaluation (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Basic usage - run experiment with automatic evaluation
    python runners/run_zs_textReason.py --benchmark complexgraph --model gpt-4 --dataset small

    # GTools benchmark
    python runners/run_zs_textReason.py --benchmark gtools --model deepseek-chat --dataset large

    # With custom output directory
    python runners/run_zs_textReason.py --benchmark complexgraph --model gpt-4 --dataset large --output_dir ./results

    # Run specific tasks only
    python runners/run_zs_textReason.py --benchmark complexgraph --model gpt-4 --dataset small --tasks clustering diameter

    # Limit test instances for quick testing
    python runners/run_zs_textReason.py --benchmark complexgraph --model gpt-4 --dataset small --max_instances 5

    # Evaluate existing results without re-running experiments
    python runners/run_zs_textReason.py --benchmark complexgraph --evaluate_only --results_file ./results/all_results.json

Output Files:
    - {task}_results.json: Per-task results with predictions
    - all_results.json: Combined raw results
    - all_results_with_eval.json: Results with evaluation details
    - evaluation_metrics.json: Accuracy and performance metrics
        """
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["complexgraph", "gtools"],
        help="Benchmark dataset to use"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4, gpt-3.5-turbo, deepseek, llama, claude-3-opus, opencoder)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset version (complexgraph: small/large/composite, gtools: small/large)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for results"
    )

    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum number of test instances per task (default: process all)"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=None,
        help="Specific task names to run (default: run all tasks)"
    )

    parser.add_argument(
        "--evaluate_only",
        action='store_true',
        help="Only evaluate existing results without running new experiments (requires --results_file)"
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to existing results JSON file for evaluation-only mode"
    )

    args = parser.parse_args()

    # Get dataset config and validate
    config = get_dataset_config(args.benchmark, args.dataset)

    # Handle evaluation-only mode
    if args.evaluate_only:
        if not args.results_file:
            print("Error: --results_file is required when using --evaluate_only")
            return

        results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"Error: Results file not found: {results_path}")
            return

        print(f"Loading existing results from: {results_path}")
        existing_data = load_json(results_path)

        if 'results' not in existing_data:
            print("Error: Invalid results file format (missing 'results' key)")
            return

        all_results = existing_data['results']
        model_name = existing_data.get('model', 'unknown')
        dataset = existing_data.get('dataset', 'unknown')

        # Evaluate
        evaluation = evaluate_all_results(all_results)

        # Save evaluation
        eval_output = results_path.parent / "evaluation_metrics.json"
        save_dict_to_json(evaluation, str(eval_output))
        print(f"✓ Evaluation metrics saved to: {eval_output}")

        # Print summary
        print_evaluation_summary(evaluation, model_name, dataset)
        return

    # Run experiment (includes evaluation)
    start_time = time.time()
    results, evaluation = run_textual_reasoning_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_instances=args.max_instances,
        task_filter=args.tasks
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")

    # Print summary report
    print_evaluation_summary(evaluation, args.model, args.dataset)


if __name__ == "__main__":
    main()
