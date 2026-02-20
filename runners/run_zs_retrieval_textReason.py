"""
Zero-Shot Retrieval-Augmented Textual Reasoning for ComplexGraph Dataset

This script augments the zero-shot textual reasoning baseline with ground-truth
retrieved documentation. The LLM receives the task description, graph structure,
AND relevant NetworkX documentation to help guide its reasoning.

Evaluation is grouped by node range: 2-5, 5-10, 10-20, 20-200, >200.

Example Usage:
    # deepseek-chat model
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model deepseek-chat --dataset small --max_instances 10 > log/ZS_RET_TEXT_output_deepseek_small.log 2>&1&
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model deepseek-chat --dataset composite --max_instances 10 > log/ZS_RET_TEXT_output_deepseek_composite.log 2>&1&
    # llama model
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model llama --dataset small --max_instances 10 > log/ZS_RET_TEXT_output_llama_small.log 2>&1&
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model llama --dataset composite --max_instances 10 > log/ZS_RET_TEXT_output_llama_composite.log 2>&1&
    # qwen-7b model
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model qwen-7b --dataset small --max_instances 10 > log/ZS_RET_TEXT_output_qwen7b_small.log 2>&1&
    nohup python runners/complexgraph/run_zs_retrieval_textReason.py --model qwen-7b --dataset composite --max_instances 10 > log/ZS_RET_TEXT_output_qwen7b_composite.log 2>&1&

Dependencies:
    - LLM API keys configured in .env file
    - ComplexGraph dataset in /data/faliwang/ComplexGraph/{small,large,composite}/
    - Retrieval ground truth: data/retrieval_groundtruth.json
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import sys
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
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

# Retrieval ground truth path
RETRIEVAL_GROUNDTRUTH_PATH = Path("data/retrieval_groundtruth.json")

# Node range bins for grouped evaluation
NODE_RANGE_BINS = [
    (2, 5, "2-5"),
    (5, 10, "5-10"),
    (10, 20, "10-20"),
    (20, 200, "20-200"),
    (200, float('inf'), ">200"),
]


def get_node_count(edge_list: List[List]) -> int:
    """Count unique nodes in an edge list."""
    nodes = set()
    for edge in edge_list:
        nodes.add(edge[0])
        nodes.add(edge[1])
    return len(nodes)


def get_node_range_label(node_count: int) -> str:
    """Map a node count to its range label."""
    for low, high, label in NODE_RANGE_BINS:
        if low <= node_count < high:
            return label
    return ">200"


def load_retrieval_groundtruth(path: Path = RETRIEVAL_GROUNDTRUTH_PATH) -> Dict[str, List[str]]:
    """Load retrieval ground truth mapping task_name -> [doc1, doc2, ...]."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded retrieval ground truth for {len(data)} tasks")
    return data


def create_retrieval_reasoning_prompt(
    question_text: str,
    edge_list: List[List],
    args: Optional[Dict[str, Any]],
    is_weighted: bool,
    is_directed: bool,
    retrieved_docs: List[str]
) -> str:
    """
    Create a prompt for retrieval-augmented textual reasoning.

    Augments the zero-shot prompt with relevant NetworkX documentation
    to help the LLM understand the graph algorithms involved.
    """
    graph_structure = format_graph_for_prompt(edge_list, is_weighted, is_directed)
    args_text = format_args_for_prompt(args)
    directed_text = "directed" if is_directed else "undirected"
    weighted_text = "weighted" if is_weighted else "unweighted"

    # Format retrieved documentation
    docs_section = ""
    if retrieved_docs:
        docs_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Truncate very long docs to keep prompt manageable
            doc_text = doc #[:2000] + "..." if len(doc) > 2000 else doc
            docs_parts.append(f"--- Document {i} ---\n{doc_text}")
        docs_section = f"""
        RELEVANT NETWORKX DOCUMENTATION:
        {chr(10).join(docs_parts)}
        """

    prompt = f"""You are an expert in graph theory and algorithms. Your task is to solve the following graph problem using textual reasoning.
{docs_section}
        TASK DESCRIPTION:
        {question_text}

        GRAPH PROPERTIES:
        - Type: {directed_text}, {weighted_text}
        - Number of edges: {len(edge_list)}

        GRAPH STRUCTURE (Edge List):
        {graph_structure}
        {args_text}

        INSTRUCTIONS:
        1. Use the provided documentation to understand the relevant algorithms
        2. Carefully analyze the graph structure provided above
        3. Apply graph theory principles to solve the problem
        4. State your final answer explicitly (e.g., "Final Answer: 42" or "Final Answer: [1, 2, 3]")

        IMPORTANT CONSTRAINTS:
        - Do NOT write any code or pseudocode
        - Base your answer purely on graph theory reasoning
        - Use the documentation as reference for algorithm definitions and behavior

        Your response: \n
        """

    return prompt


def print_node_range_evaluation(all_results: list, graphs_data: dict):
    """Print evaluation results grouped by node range."""
    # Collect per-instance results with node counts
    range_stats = {label: {'correct': 0, 'incorrect': 0, 'error': 0, 'total': 0}
                   for _, _, label in NODE_RANGE_BINS}

    for task_result in all_results:
        for pred in task_result.get('predictions', []):
            graph_id = pred.get('graph_id')
            if graph_id and graph_id in graphs_data:
                node_count = get_node_count(graphs_data[graph_id]['graph'])
                label = get_node_range_label(node_count)

                range_stats[label]['total'] += 1

                if 'error' in pred:
                    range_stats[label]['error'] += 1
                elif 'extracted_answer' in pred:
                    is_correct, _ = compare_results_type_aware(
                        pred['extracted_answer'], pred['ground_truth']
                    )
                    if is_correct:
                        range_stats[label]['correct'] += 1
                    else:
                        range_stats[label]['incorrect'] += 1

    # Print table
    print(f"\n{'='*70}")
    print(f"  Evaluation by Node Range")
    print(f"{'='*70}")
    print(f"{'Node Range':<12} {'Total':>7} {'Correct':>9} {'Incorrect':>11} {'Error':>7} {'Accuracy':>10}")
    print("-" * 70)

    overall_correct = 0
    overall_total = 0

    for _, _, label in NODE_RANGE_BINS:
        stats = range_stats[label]
        total = stats['total']
        correct = stats['correct']
        incorrect = stats['incorrect']
        error = stats['error']
        acc = (correct / total * 100) if total > 0 else 0.0

        overall_correct += correct
        overall_total += total

        if total > 0:
            print(f"{label:<12} {total:>7} {correct:>9} {incorrect:>11} {error:>7} {acc:>9.1f}%")
        else:
            print(f"{label:<12} {total:>7} {'-':>9} {'-':>11} {'-':>7} {'N/A':>10}")

    print("-" * 70)
    overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
    print(f"{'Overall':<12} {overall_total:>7} {overall_correct:>9} {'':>11} {'':>7} {overall_acc:>9.1f}%")
    print(f"{'='*70}\n")

    return range_stats


def run_retrieval_textual_reasoning_experiment(
    model_name: str,
    dataset_version: str,
    output_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None
):
    """
    Run retrieval-augmented textual reasoning experiment on ComplexGraph dataset.
    """
    print(f"\n{'='*70}")
    print(f"Retrieval-Augmented Textual Reasoning Experiment")
    print(f"Model: {model_name}")
    print(f"Dataset: ComplexGraph-{dataset_version.upper()}")
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

    # Load retrieval ground truth
    try:
        retrieval_gt = load_retrieval_groundtruth()
    except Exception as e:
        print(f"✗ Error loading retrieval ground truth: {e}")
        return

    # Load dataset
    try:
        questions_data, graphs_data = load_complexgraph_data(dataset_version)
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Set up output directory
    if output_dir is None:
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        output_dir = Path("LLM_generation_results") / "complexgraph" / "textual_reasoning" / "retrieval_zero_shot" / dataset_version / model_short
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

        # Get retrieved docs for this task
        retrieved_docs = retrieval_gt.get(task_name, [])

        print(f"\n{'='*60}")
        print(f"Question Group {group_idx + 1}/{len(questions_data)}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"Retrieved docs: {len(retrieved_docs)}")
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        # Limit instances if specified
        instances_to_process = graph_instances[:max_instances] if max_instances else graph_instances

        task_results = {
            'task_name': task_name,
            'weighted': is_weighted,
            'directed': is_directed,
            'question_text': question_text,
            'num_retrieved_docs': len(retrieved_docs),
            'predictions': []
        }

        for inst_idx, instance in enumerate(tqdm(instances_to_process, desc=f"Processing {task_name}")):
            graph_id = instance['graph']
            args = instance['args']
            ground_truth = instance['answer']

            if graph_id not in graphs_data:
                print(f"⚠ Warning: Graph {graph_id} not found in graphs.json")
                continue

            graph_info = graphs_data[graph_id]
            edge_list = graph_info['graph']
            node_count = get_node_count(edge_list)

            # Create retrieval-augmented prompt
            prompt = create_retrieval_reasoning_prompt(
                question_text=question_text,
                edge_list=edge_list,
                args=args,
                is_weighted=is_weighted,
                is_directed=is_directed,
                retrieved_docs=retrieved_docs
            )

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
                    'extracted_answer': extracted_answer,
                    'prompt_length': len(prompt),
                    'node_count': node_count,
                    'node_range': get_node_range_label(node_count)
                })

            except Exception as e:
                print(f"\n✗ Error processing instance {inst_idx}: {e}")
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': str(e),
                    'node_count': node_count,
                    'node_range': get_node_range_label(node_count)
                })

        # Calculate task-level statistics
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

            task_results['statistics'] = {
                'total_instances': total_instances,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'error_count': error_count,
                'correct_rate': correct_rate,
                'incorrect_rate': incorrect_rate,
                'error_rate': error_rate
            }

        all_results.append(task_results)

        # Save intermediate results
        output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        save_dict_to_json(task_results, str(output_file))
        print(f"✓ Saved results to: {output_file}")

    # Save combined results
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version},
                      str(combined_output))
    print(f"✓ Raw results saved to: {combined_output}")

    # Standard evaluation
    evaluation_results = evaluate_all_results(all_results)

    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}")

    # Node-range grouped evaluation
    range_stats = print_node_range_evaluation(all_results, graphs_data)

    # Save results with all evaluations
    combined_with_eval = output_dir / "all_results_with_eval.json"
    save_dict_to_json({
        'results': all_results,
        'evaluation': evaluation_results,
        'evaluation_by_node_range': range_stats,
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
    parser = argparse.ArgumentParser(
        description="Retrieval-augmented zero-shot textual reasoning on ComplexGraph dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    python runners/complexgraph/run_zs_retrieval_textReason.py --model deepseek-chat --dataset small
    python runners/complexgraph/run_zs_retrieval_textReason.py --model llama --dataset composite --max_instances 10
    python runners/complexgraph/run_zs_retrieval_textReason.py --model qwen-7b --dataset large --tasks clustering diameter
    python runners/complexgraph/run_zs_retrieval_textReason.py --evaluate_only --results_file ./results/all_results.json --dataset small
        """
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., deepseek-chat, llama, qwen-7b)")
    parser.add_argument("--dataset", type=str, required=True, choices=["small", "large", "composite"],
                        help="ComplexGraph dataset version")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Maximum instances per task")
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                        help="Specific task names to run")
    parser.add_argument("--evaluate_only", action='store_true',
                        help="Only evaluate existing results")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to existing results JSON for evaluation-only mode")

    args = parser.parse_args()

    # Handle evaluation-only mode
    if args.evaluate_only:
        if not args.results_file:
            print("Error: --results_file is required when using --evaluate_only")
            return
        if not args.dataset:
            print("Error: --dataset is required for node-range evaluation")
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
        dataset = existing_data.get('dataset', args.dataset)

        # Load graphs for node-range evaluation
        _, graphs_data = load_complexgraph_data(dataset)

        # Standard evaluation
        evaluation = evaluate_all_results(all_results)
        eval_output = results_path.parent / "evaluation_metrics.json"
        save_dict_to_json(evaluation, str(eval_output))
        print(f"✓ Evaluation metrics saved to: {eval_output}")

        # Node-range evaluation
        print_node_range_evaluation(all_results, graphs_data)

        print_evaluation_summary(evaluation, model_name, dataset)
        return

    # Run experiment
    start_time = time.time()
    results, evaluation = run_retrieval_textual_reasoning_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_instances=args.max_instances,
        task_filter=args.tasks
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")

    print_evaluation_summary(evaluation, args.model, args.dataset)


if __name__ == "__main__":
    main()
