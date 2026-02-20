"""
Shared utility functions for ComplexGraph textual reasoning experiments.

This module contains common functions used by both zero-shot and few-shot
textual reasoning scripts for the ComplexGraph dataset.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import re, ast

COMPLEXGRAPH_BASE_PATH = Path("/data/faliwang/GTools")


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Loaded JSON data as dict, or empty dict if file not found/invalid
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠ Warning: Could not load {path}: {e}")
        return {}


def load_complexgraph_data(dataset_version: str, base_path: Path = None):
    """
    Load ComplexGraph dataset in the new format.

    The new format has:
    - questions.json: List of question groups, each with task_name, weighted, directed,
                     question variants, and graph_data list
    - graphs.json: Dict of {graph_id: {graph, weighted, directed, generation_type,
                   connected, cyclic}}

    Args:
        dataset_version: One of 'small', 'large', 'composite'
        base_path: Base directory containing ComplexGraph data

    Returns:
        Tuple of (questions_data, graphs_data)
    """
    if base_path is None:
        base_path = Path(COMPLEXGRAPH_BASE_PATH)

    dataset_dir = base_path / dataset_version

    # Load questions (list of question groups)
    questions_path = dataset_dir / "questions.json"
    questions_data = load_json(questions_path)

    # Load graphs (dict of graph_id -> graph_data)
    graphs_path = dataset_dir / "graphs.json"
    graphs_data = load_json(graphs_path)

    if not questions_data or not graphs_data:
        raise ValueError(f"Failed to load data from {dataset_dir}")

    print(f"✓ Loaded {len(questions_data)} question groups and {len(graphs_data)} graphs")

    return questions_data, graphs_data


def format_graph_for_prompt(edge_list: List[List], is_weighted: bool, is_directed: bool) -> str:
    """
    Format edge list into natural language for the prompt.

    Args:
        edge_list: List of edges, each edge is [source, target] or [source, target, weight]
        is_weighted: Whether the graph is weighted
        is_directed: Whether the graph is directed

    Returns:
        Formatted string describing the graph structure
    """
    edge_word = "directed edge" if is_directed else "undirected edge"
    edges_formatted = []

    for edge in edge_list:
        if is_weighted and len(edge) == 3:
            edges_formatted.append(
                f"  {edge_word.capitalize()} from node {edge[0]} to node {edge[1]} with weight {edge[2]:.4f}"
            )
        else:
            # Unweighted edge
            edges_formatted.append(
                f"  {edge_word.capitalize()} from node {edge[0]} to node {edge[1]}"
            )

    return "\n".join(edges_formatted)


def format_args_for_prompt(args: Optional[Dict[str, Any]]) -> str:
    """
    Format task arguments for the prompt.

    Args:
        args: Dictionary of arguments (e.g., {"node": 5} or {"source": 1, "target": 3})
              Can be None for tasks without arguments

    Returns:
        Formatted string describing the arguments, or empty string if no args
    """
    if not args:
        return ""

    args_parts = []
    for key, value in args.items():
        args_parts.append(f"{key} = {value}")

    return "\n\nTask Parameters:\n" + "\n".join(f"  {part}" for part in args_parts)


def extract_answer_from_response(response: str) -> str:
    """
    Extract the final answer from LLM response.

    Looks for patterns like "Final Answer: X" or "Answer: X" and extracts X.
    If no clear pattern is found, returns the last line or full response.

    Args:
        response: Raw LLM response text

    Returns:
        Extracted answer string
    """
    response = response.strip()

    # Look for explicit answer markers
    markers = ["Final Answer:", "Final answer:", "Answer:", "answer:",
               "The answer is", "Therefore,"]

    for marker in markers:
        if marker in response:
            # Get text after the marker
            answer_part = response.split(marker)[-1].strip()
            # Take first line after marker (often the cleanest)
            answer_line = answer_part.split('\n')[0].strip()
            if answer_line:
                return answer_line

    # If no marker found, try to extract the last substantive line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return response


def clean_extracted_answer(raw: str) -> str:
    """
    Clean extracted_answer string by removing markdown artifacts and whitespace.

    Handles patterns like:
        "0.16666666666666666**"  -> "0.16666666666666666"
        "** 0.16666666666666666" -> "0.16666666666666666"
        "false**"               -> "false"
        "** false"              -> "false"
        "[65]**"                -> "[65]"
        "True**"                -> "True"
    """
    s = raw.strip()
    # Remove leading/trailing ** (markdown bold markers)
    s = s.strip('*')
    s = s.strip()
    # Remove any remaining markdown formatting like \boxed{...}
    boxed_match = re.search(r'\\boxed\{(.+?)\}', s)
    if boxed_match:
        s = boxed_match.group(1).strip()
    return s


def parse_answer(answer_str: str, ground_truth: Any) -> Any:
    """
    Parse answer string to appropriate type based on ground truth.

    Attempts to convert the extracted answer string to the same type as the ground truth
    for fair comparison. Handles numbers, lists, booleans, and strings.

    Args:
        answer_str: Extracted answer string from LLM
        ground_truth: Ground truth value (determines expected type)

    Returns:
        Parsed answer in appropriate type, or original string if parsing fails
    """
    import re
    import ast

    if answer_str is None:
        return None

    answer_str = str(answer_str).strip()

    # Handle boolean ground truth
    if isinstance(ground_truth, bool):
        answer_lower = answer_str.lower()
        if 'true' in answer_lower or answer_lower == '1':
            return True
        elif 'false' in answer_lower or answer_lower == '0':
            return False
        return answer_str

    # Handle list ground truth
    if isinstance(ground_truth, (list, tuple)):
        # Try to find and parse a list pattern [...]
        list_match = re.search(r'\[([^\]]*)\]', answer_str)
        if list_match:
            try:
                parsed_list = ast.literal_eval(list_match.group(0))
                return parsed_list
            except:
                pass
        return answer_str

    # Handle numeric ground truth (int or float)
    if isinstance(ground_truth, (int, float)):
        # Extract first number from answer
        num_match = re.search(r'-?\d+\.?\d*', answer_str)
        if num_match:
            num_str = num_match.group(0)
            try:
                if isinstance(ground_truth, int):
                    return int(float(num_str))
                else:
                    return float(num_str)
            except:
                pass
        return answer_str

    # Handle None ground truth (no parsing needed)
    if ground_truth is None:
        return answer_str

    # Default: return as string
    return answer_str


def parse_answer_text(cleaned: str, ground_truth: Any) -> Any:
    """
    Parse a cleaned answer string into the appropriate Python type.

    Uses ground_truth type as a hint for parsing.
    Returns the parsed value, or the original string if parsing fails.
    """
    if cleaned == "" or cleaned is None:
        return None

    # Try boolean parsing first (before numeric, since "true"/"false" are common)
    lower = cleaned.lower().strip()

    # Semantic equivalence: yes/no -> True/False
    if lower in ('true', 'yes'):
        return True
    if lower in ('false', 'no'):
        return False

    # Try list parsing: "[65]", "[20, 25]"
    if cleaned.startswith('['):
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    # Try numeric parsing
    try:
        # Try int first
        if '.' not in cleaned and 'e' not in cleaned.lower():
            return int(cleaned)
        else:
            return float(cleaned)
    except ValueError:
        pass

    # Return original string if nothing worked
    return cleaned


def compare_results_type_aware(predicted, ground_truth) -> Tuple[bool, str]:
    """
    Compare predicted result with ground truth using type-aware comparison rules.

    This function handles different data types appropriately:
    - Integers and booleans: Direct equality
    - Floats: Rounded to 2 decimal places for comparison
    - Lists: Sorted comparison (order doesn't matter for most graph tasks)
    - Mixed int/float: Absolute difference < 1
    - None and empty list: Special equivalence
    - Tuples: Extract first element for some tasks

    Args:
        predicted: Result from code execution
        ground_truth: Expected correct answer

    Returns:
        Tuple[bool, str]: (is_correct, reason)
            - is_correct: Whether the prediction matches ground truth
            - reason: Explanation of the comparison result

    Examples:
        >>> compare_results_type_aware(5, 5)
        (True, "Exact match (int/bool)")

        >>> compare_results_type_aware(3.14159, 3.14)
        (True, "Match after rounding to 2 decimals")

        >>> compare_results_type_aware([1, 2, 3], [3, 1, 2])
        (True, "Match after sorting")

        >>> compare_results_type_aware(5, 5.0)
        (True, "Match with type conversion (int/float difference < 1)")
    """
    # Case 1: Both are integers or both are booleans - direct comparison
    if (isinstance(ground_truth, int) and isinstance(predicted, int)) or \
       (isinstance(ground_truth, bool) and isinstance(predicted, bool)):
        is_correct = str(ground_truth) == str(predicted)
        reason = "Exact match (int/bool)" if is_correct else f"Mismatch: {predicted} != {ground_truth}"
        return is_correct, reason

    # Case 2: Both are floats - compare with rounding to 2 decimal places
    # This handles floating-point precision issues
    elif isinstance(ground_truth, float) and isinstance(predicted, float):
        is_correct = round(ground_truth, 2) == round(predicted, 2)
        reason = "Match after rounding to 2 decimals" if is_correct else \
                 f"Mismatch: {round(predicted, 2)} != {round(ground_truth, 2)}"
        return is_correct, reason

    # Case 3: Both are lists - compare after sorting
    # Order doesn't matter for most graph tasks (e.g., list of nodes)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        is_correct = sorted(ground_truth) == sorted(predicted)
        reason = "Match after sorting" if is_correct else \
                 f"Mismatch: {sorted(predicted)} != {sorted(ground_truth)}"
        return is_correct, reason

    # Case 4: Mixed int and float - check if difference is small
    # Handles cases where result is 5.0 but expected is 5
    elif (isinstance(ground_truth, float) and isinstance(predicted, int)) or \
         (isinstance(ground_truth, int) and isinstance(predicted, float)):
        is_correct = abs(ground_truth - predicted) < 1
        reason = "Match with type conversion (int/float difference < 1)" if is_correct else \
                 f"Mismatch: |{predicted} - {ground_truth}| >= 1"
        return is_correct, reason

    # Case 5: Same types - direct comparison
    elif type(ground_truth) == type(predicted):
        is_correct = ground_truth == predicted
        reason = f"Exact match ({type(ground_truth).__name__})" if is_correct else \
                 f"Mismatch: {predicted} != {ground_truth}"
        return is_correct, reason

    # Case 6: None vs empty list - special equivalence
    # Some tasks return None for empty results, others return []
    elif ground_truth is None and isinstance(predicted, list):
        is_correct = len(predicted) == 0
        reason = "Match (None equivalent to empty list)" if is_correct else \
                 f"Mismatch: expected None, got non-empty list {predicted}"
        return is_correct, reason

    # Case 7: Tuple result vs float ground truth
    # Some algorithms return (value, extra_info) tuples
    elif isinstance(predicted, tuple) and isinstance(ground_truth, float):
        is_correct = abs(predicted[0] - ground_truth) < 1
        reason = "Match (extracted first element from tuple)" if is_correct else \
                 f"Mismatch: |{predicted[0]} - {ground_truth}| >= 1"
        return is_correct, reason

    # Case 8: Type mismatch - cannot compare
    else:
        return False, f"Type mismatch: predicted is {type(predicted).__name__}, " \
                     f"ground_truth is {type(ground_truth).__name__}"


def calculate_task_metrics(task_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a single task.

    Computes accuracy, error counts, and other statistics for predictions
    on a specific task.

    Args:
        task_results: Task results dict with 'predictions' list

    Returns:
        Dict containing accuracy, counts, and detailed statistics
    """
    predictions = task_results.get('predictions', [])

    if not predictions:
        return {
            'total_instances': 0,
            'accuracy': 0.0,
            'error': 'No predictions found'
        }

    total = len(predictions)
    correct = 0
    errors = 0
    no_ground_truth = 0
    match_type_counts = {}

    for pred in predictions:
        if 'error' in pred:
            errors += 1
            continue

        ground_truth = pred.get('ground_truth')
        extracted = pred.get('extracted_answer')

        # Parse answer to appropriate type
        parsed_answer = parse_answer(extracted, ground_truth)

        # Evaluate
        eval_result = compare_results_type_aware(parsed_answer, ground_truth)

        # Store evaluation in prediction
        pred['parsed_answer'] = parsed_answer
        pred['evaluation'] = eval_result

        if eval_result['match_type'] == 'no_ground_truth':
            no_ground_truth += 1
        elif eval_result['correct']:
            correct += 1

        # Count match types
        match_type = eval_result['match_type']
        match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1

    # Calculate metrics
    valid_instances = total - errors - no_ground_truth
    accuracy = (correct / total * 100) if total > 0 else 0.0

    metrics = {
        'total_instances': total,
        'valid_instances': valid_instances,
        'correct_predictions': correct,
        'incorrect_predictions': total - correct,
        'errors': errors,
        'no_ground_truth': no_ground_truth,
        'accuracy': accuracy,
        'match_type_distribution': match_type_counts
    }

    return metrics

def evaluate_prediction():
    pass

def evaluate_all_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate all task results and compute aggregate metrics.

    Processes results from all tasks, calculates per-task and overall metrics,
    and generates a comprehensive evaluation report.

    Args:
        all_results: List of task result dicts from experiment

    Returns:
        Dict containing per-task metrics, overall metrics, and summary statistics
    """
    print(f"\n{'='*70}")
    print("EVALUATING RESULTS")
    print(f"{'='*70}\n")

    task_metrics = {}
    overall_correct = 0
    overall_valid_total = 0
    overall_total = 0

    for task_result in all_results:
        task_name = task_result['task_name']
        is_weighted = task_result.get('weighted')
        is_directed = task_result.get('directed')
        take_name_weighted_directed = f"{task_name}_weighted_{is_weighted}_directed_{is_directed}"
        print(f"Evaluating task: {task_name}")

        # Calculate metrics for this task
        # metrics = calculate_task_metrics(task_result)
        metrics = task_result['statistics']
        task_metrics[take_name_weighted_directed] = metrics

        # Update overall counts
        overall_correct += metrics.get('correct_count', 0)
        overall_total += metrics.get('total_instances', 0)
        overall_valid_total += (overall_total - metrics.get('error_count', 0))

        # Print task summary
        print(f"  Accuracy: {metrics['correct_rate']:.2f}% ({metrics['correct_count']}/{metrics['total_instances']})")
        if metrics['error_count'] > 0:
            print(f"  Errors: {metrics['error_count']}")

    # Calculate overall metrics
    overall_valid_accuracy = (overall_correct / overall_valid_total * 100) if overall_valid_total > 0 else 0.0
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0

    overall_metrics = {
        'total_tasks': len(all_results),
        'total_instances': overall_total,
        'total_correct': overall_correct,
        'overall_valid_accuracy': overall_valid_accuracy,
        'overall_accuracy': overall_accuracy
    }

    print(f"\n{'='*70}")
    print("OVERALL METRICS")
    print(f"{'='*70}")
    print(f"Total tasks: {overall_metrics['total_tasks']}")
    print(f"Total instances: {overall_metrics['total_instances']}")
    print(f"Overall accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")
    print(f"{'='*70}\n")

    return {
        'task_metrics': task_metrics,
        'overall_metrics': overall_metrics,
        'per_task_accuracy': {task: metrics['correct_rate'] for task, metrics in task_metrics.items()}
    }


def print_evaluation_summary(evaluation: Dict[str, Any], model_name: str, dataset: str):
    """
    Print a formatted summary of evaluation results.

    Args:
        evaluation: Evaluation results dict
        model_name: Name of the model used
        dataset: Dataset version
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"{'='*70}\n")

    # Overall metrics
    overall = evaluation['overall_metrics']
    print(f"Overall Performance:")
    print(f"  Total tasks evaluated: {overall['total_tasks']}")
    print(f"  Total instances: {overall['total_instances']}")
    print(f"  Correct predictions: {overall['total_correct']}")
    print(f"  Overall accuracy: {overall['overall_accuracy']:.2f}%")

    # Per-task breakdown
    print(f"\nPer-Task Accuracy:")
    print(f"{'-'*70}")
    per_task = evaluation['per_task_accuracy']

    # Sort by accuracy for better readability
    sorted_tasks = sorted(per_task.items(), key=lambda x: x[1], reverse=True)

    for task_name, accuracy in sorted_tasks:
        task_metrics = evaluation['task_metrics'][task_name]
        correct = task_metrics['correct_count']
        total = task_metrics['total_instances']
        print(f"  {task_name:40s}: {accuracy:6.2f}% ({correct:4d}/{total:4d})")

    # Identify best and worst performing tasks
    if sorted_tasks:
        best_task, best_acc = sorted_tasks[0]
        worst_task, worst_acc = sorted_tasks[-1]

        print(f"\nBest performing task:")
        print(f"  {best_task}: {best_acc:.2f}%")

        print(f"\nWorst performing task:")
        print(f"  {worst_task}: {worst_acc:.2f}%")

    print(f"\n{'='*70}\n")
