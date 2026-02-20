"""
CodeGraph Baseline: Code Generation with Provided Examples (Unified)

This script generates code using pre-defined examples as guidance, then evaluates
the generated code on test instances.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_codegraph.py --benchmark complexgraph --model deepseek-chat --dataset small --max_instances 10
    python runners/run_codegraph.py --benchmark complexgraph --model llama --dataset composite --max_instances 10

    # GTools
    python runners/run_codegraph.py --benchmark gtools --model deepseek-chat --dataset small --max_instances 10
    python runners/run_codegraph.py --benchmark gtools --model llama --dataset large --max_instances 10

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark
    - Code examples in prompts/codegraph_examples/task_examples.json

Output Files:
    - {task}_results.json: Per-task results with predictions
    - all_results.json: Combined results from all tasks
    - all_results_with_eval.json: Results with evaluation annotations
    - evaluation_metrics.json: Accuracy metrics and statistics
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import load_env
    print("✓ API keys loaded from .env file")
except ImportError:
    print("⚠ Warning: load_env not found, ensure API keys are set manually")

from utils.get_llm_response_generator import create_code_generator
from utils.shared.json_utils import save_dict_to_json
from utils.code_execution_utils import (
    generate_code_with_llm,
    extract_code_from_response,
    execute_code_with_timeout,
    compare_results_type_aware
)
from utils.complexgraph_utils import (
    evaluate_all_results,
    print_evaluation_summary
)
from utils.dataset_config import get_dataset_config

# CODE_EXAMPLES_FILE = Path("prompts/codegraph_examples/task_examples.json")
CODE_EXAMPLES_FILE = Path("data/retrieval_groundtruth.json")


def get_code_example_for_task(code_examples: List[Dict], task_name: str) -> Optional[str]:
    """Get code example for a specific task."""
    # for example in code_examples:
    #     if example.get("function") == task_name:
    #         return example.get("signature")
    if task_name in code_examples: 
        return "\n\n".join(code_examples[task_name])
    return None


def run_codegraph_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    task_filter: Optional[List[str]] = None,
    max_instances: Optional[int] = None
):
    """
    Run CodeGraph experiment with code generation and evaluation.

    Args:
        model_name: Name of the model to run
        dataset_version: Dataset version identifier
        config: Dataset configuration from get_dataset_config()
        output_dir: Custom output directory
        task_filter: Optional list of task names to run
        max_instances: Maximum test instances per task
    """
    print(f"\n{'='*70}")
    print(f"CodeGraph - Code Generation with Provided Examples")
    print(f"Model: {model_name}")
    print(f"Dataset: {config['label']}-{dataset_version.upper()}")
    if max_instances:
        print(f"Max instances per task: {max_instances}")
    if task_filter:
        print(f"Task filter: {task_filter}")
    print(f"{'='*70}\n")

    # Set adaptive timeout based on dataset size
    if dataset_version == "large":
        execution_timeout = 2000
    else:
        execution_timeout = 30
    print(f"Execution timeout: {execution_timeout} seconds")

    if not CODE_EXAMPLES_FILE.exists():
        print(f"✗ Error: Code examples file not found: {CODE_EXAMPLES_FILE}")
        return

    with open(CODE_EXAMPLES_FILE, "r") as f:
        code_examples = json.load(f)
    print(f"✓ Loaded {len(code_examples)} code examples\n")

    try:
        questions_data, graphs_data = config["load_data"](dataset_version)
        print(f"✓ Loaded {config['label']} dataset: {dataset_version}")
        print(f"  - Questions: {len(questions_data)} groups")
        print(f"  - Graphs: {len(graphs_data)} instances\n")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    if output_dir is None:
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        output_dir = (
            Path("LLM_generation_results") / config["output_base"] / "code_generation" /
            "codegraph" / dataset_version / model_short
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        llm_generator = create_code_generator(model_name)
        print(f"✓ LLM initialized\n")
    except Exception as e:
        print(f"✗ Error initializing LLM: {e}\n")
        return

    all_results = []

    for group_idx, question_group in enumerate(questions_data):
        task_name = question_group['task_name']
        is_weighted = question_group['weighted']
        is_directed = question_group['directed']
        graph_instances = question_group['graph_data']

        if task_filter and task_name not in task_filter:
            print(f"⊘ Skipping {task_name} (not in task filter)")
            continue

        print(f"\n{'='*60}")
        print(f"Question Group {group_idx + 1}/{len(questions_data)}")
        print(f"Task: {task_name}")
        print(f"Properties: directed={is_directed}, weighted={is_weighted}")
        print(f"Instances: {len(graph_instances)}")
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        code_example = get_code_example_for_task(code_examples, task_name)
        if code_example:
            print(f"[1/3] ✓ Code example found")
        else:
            print(f"[1/3] ⚠ No code example (zero-shot)")

        sample_args = graph_instances[0]['args'] if graph_instances else None
        args_desc = ""
        if sample_args:
            args_desc = "and " + ", ".join(sample_args.keys())

        weighted_str = "weighted" if is_weighted else "unweighted"
        directed_str = "directed" if is_directed else "undirected"

        prompt = (
            f"Given the task description: {question_text}, "
            f"generate a python function that take an edge list of {weighted_str} {directed_str} graph "
            f"in Python datatype List[List[Float]] {args_desc} as input"
            f". Return Python datatype None if any error occurs"
        )

        print(f"\n[2/3] Generating code with code examples...")
        try:
            code_response = generate_code_with_llm(
                query=prompt,
                llm_model=llm_generator,
                retrieved_docs=code_example
            )
            tokens_code_prompt = len(prompt.split())
            print(f"  [LLM] Code generation prompt sent ({tokens_code_prompt} tokens)")
            tokens_code_response = len(code_response.split())
            print(f"  [LLM] Code generation response received ({tokens_code_response} tokens)")
            generated_code = extract_code_from_response(code_response)
            print(f"✓ Code generated ({len(generated_code)} chars)\n")
        except Exception as e:
            print(f"✗ Error generating code: {e}\n")
            continue

        print(f"[3/3] Executing code on {len(graph_instances)} instances...")

        instances_to_process = graph_instances[:max_instances] if max_instances else graph_instances

        task_results = {
            'task_name': task_name,
            'weighted': is_weighted,
            'directed': is_directed,
            'question_text': question_text,
            'generated_code': generated_code,
            'code_example_used': code_example if code_example else None,
            'predictions': []
        }

        correct_count = 0
        error_count = 0
        incorrect_count = 0

        for inst_idx, instance in enumerate(tqdm(instances_to_process, desc=f"Executing {task_name}")):
            graph_id = instance['graph']
            args = instance['args']
            ground_truth = instance['answer']

            if graph_id not in graphs_data:
                print(f"⚠ Warning: Graph {graph_id} not found in graphs.json")
                error_count += 1
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': 'Graph not found in dataset'
                })
                continue

            graph_info = graphs_data[graph_id]
            edge_list = graph_info['graph']

            try:
                result = execute_code_with_timeout(
                    code=generated_code,
                    edge_list=edge_list,
                    args=args,
                    timeout_seconds=execution_timeout
                )

                is_correct, comparison_reason = compare_results_type_aware(result, ground_truth)

                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'execution_result': result,
                    'is_correct': is_correct,
                    'comparison_reason': comparison_reason,
                    'extracted_answer': result
                })

            except TimeoutError as e:
                error_count += 1
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': f'Timeout: {str(e)}',
                    'error_type': 'timeout'
                })

            except Exception as e:
                error_type = e.__class__.__name__
                error_count += 1
                task_results['predictions'].append({
                    'instance_id': inst_idx,
                    'graph_id': graph_id,
                    'args': args,
                    'ground_truth': ground_truth,
                    'error': str(e),
                    'error_type': error_type
                })

        print(f"\nTask Statistics:")
        total_instances = len(instances_to_process)
        if total_instances > 0:
            correct_rate = correct_count / total_instances
            error_rate = error_count / total_instances
            incorrect_rate = incorrect_count / total_instances

            print(f"{'='*60}")
            print(f"Results for {task_name}:")
            print(f"  Total instances: {total_instances}")
            print(f"  ✓ Correct: {correct_count} ({correct_rate*100:.1f}%)")
            print(f"  ✗ Incorrect: {incorrect_count} ({incorrect_rate*100:.1f}%)")
            print(f"  ⚠ Errors: {error_count} ({error_rate*100:.1f}%)")
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

        output_file = output_dir / f"{task_name}_weighted_{is_weighted}_directed_{is_directed}_results.json"
        print("### Debug task_results: ", type(task_results))
        save_dict_to_json(task_results, str(output_file))
        print(f"✓ Saved results to: {output_file}")

    print(f"\n{'='*70}")
    print(f"Finalizing Results...")
    print(f"{'='*70}\n")

    combined_output = output_dir / "all_results.json"
    save_dict_to_json({
        'results': all_results,
        'model': model_name,
        'dataset': dataset_version,
        'code_examples_file': str(CODE_EXAMPLES_FILE)
    }, str(combined_output))
    print(f"✓ Raw results saved to: {combined_output}")

    for task_result in all_results:
        for pred in task_result['predictions']:
            if 'execution_result' in pred and 'extracted_answer' not in pred:
                pred['extracted_answer'] = pred.get('execution_result')

    evaluation_results = evaluate_all_results(all_results)

    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}")

    combined_with_eval = output_dir / "all_results_with_eval.json"
    save_dict_to_json({
        'results': all_results,
        'evaluation': evaluation_results,
        'model': model_name,
        'dataset': dataset_version,
        'code_examples_file': str(CODE_EXAMPLES_FILE)
    }, str(combined_with_eval))
    print(f"✓ Results with evaluation saved to: {combined_with_eval}")

    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"Results directory: {output_dir}")
    print(f"Question groups processed: {len(all_results)}")
    print(f"Overall accuracy: {evaluation_results['overall_metrics']['overall_accuracy']:.2f}%")
    print(f"{'='*70}\n")

    print_evaluation_summary(evaluation_results, model_name, dataset_version)

    del llm_generator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CodeGraph: Code Generation with Provided Examples (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Run with single model
    python runners/run_codegraph.py --benchmark complexgraph --model deepseek-chat --dataset small

    # GTools benchmark
    python runners/run_codegraph.py --benchmark gtools --model deepseek-chat --dataset large

    # Run specific tasks
    python runners/run_codegraph.py --benchmark complexgraph --model llama --dataset small --tasks clustering diameter

    # Quick testing with limited instances
    python runners/run_codegraph.py --benchmark complexgraph --model llama --dataset small --max_instances 10

Output:
    - {task}_results.json: Per-task results with predictions
    - all_results.json: Combined raw results
    - all_results_with_eval.json: Results with evaluation details
    - evaluation_metrics.json: Accuracy and performance metrics
        """
    )

    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["complexgraph", "gtools"],
                        help="Benchmark dataset to use")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset version (complexgraph: small/large/composite, gtools: small/large)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to run")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Maximum number of test instances per task")

    args = parser.parse_args()

    # Get dataset config and validate
    config = get_dataset_config(args.benchmark, args.dataset)

    start_time = time.time()
    run_codegraph_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        task_filter=args.tasks,
        max_instances=args.max_instances
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()
