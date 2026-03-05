"""
Zero-Shot Sentence-BERT + Coding Agent (Unified)

This script implements retrieval-augmented code generation using Sentence-BERT
embeddings to retrieve relevant documentation, combined with iterative error correction.

Supports both ComplexGraph and GTools benchmarks via --benchmark flag.

Example Usage:
    # ComplexGraph
    python runners/run_zs_sentbert_codingagent.py --benchmark complexgraph --model deepseek-chat --dataset small
    python runners/run_zs_sentbert_codingagent.py --benchmark complexgraph --model llama --dataset composite

    # GTools
    python runners/run_zs_sentbert_codingagent.py --benchmark gtools --model deepseek-chat --dataset small
    python runners/run_zs_sentbert_codingagent.py --benchmark gtools --model llama --dataset large

Dependencies:
    - LLM API keys configured in .env file
    - Dataset files for the chosen benchmark
    - Sentence-BERT embeddings for documentation retrieval

Output Files:
    - {task_name}_results.json: Per-task results with generated code
    - all_results.json: Combined results from all tasks
    - evaluation_metrics.json: Accuracy metrics
"""

import argparse
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
    create_error_correction_prompt,
    load_test_case_from_file
)
from utils.dataset_config import get_dataset_config

# Sentence-BERT retrieval
try:
    from utils.sentbert_retrieval import find_best_docstring
    HAS_SENTBERT = True
except ImportError:
    HAS_SENTBERT = False
    print("⚠ Warning: Sentence-BERT retriever not available")


def run_sentbert_coding_experiment(
    model_name: str,
    dataset_version: str,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    max_correction_rounds: int = 3,
    max_instances: Optional[int] = None,
    task_filter: Optional[List[str]] = None
):
    """
    Run Sentence-BERT retrieval + coding agent experiment.

    Args:
        model_name: Name of LLM model
        dataset_version: Dataset version
        config: Dataset configuration from get_dataset_config()
        output_dir: Custom output directory
        max_correction_rounds: Maximum error correction iterations
        max_instances: Maximum test instances per task
        task_filter: List of specific task names to run
    """
    print(f"\n{'='*70}")
    print(f"Sentence-BERT + Coding Agent Experiment")
    print(f"Model: {model_name}")
    print(f"Dataset: {config['label']}-{dataset_version.upper()}")
    print(f"{'='*70}\n")

    if not HAS_SENTBERT:
        print("✗ Sentence-BERT retriever not available")
        return

    # Initialize retriever
    try:
        docs_repo_path = config.get("docs_repo", "data/networkx_graph_functions_docs.json")

        class SentenceBertRetriever:
            """Compatibility adapter around find_best_docstring()."""

            def __init__(self, repo_json_path: str):
                self.repo_json_path = repo_json_path

            def retrieve(self, query_text: str, top_k: int = 3):
                return find_best_docstring(
                    repo_json_path=self.repo_json_path,
                    query_text=query_text,
                    top_k=top_k,
                )

        retriever = SentenceBertRetriever(repo_json_path=docs_repo_path)
        print(f"✓ Sentence-BERT retriever initialized\n")
    except Exception as e:
        print(f"✗ Error initializing retriever: {e}")
        return

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
        output_dir = Path("LLM_generation_results") / config["output_base"] / "code_generation" / "sentbert_codingagent" / dataset_version / model_short
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"{'='*60}\n")

        question_text = question_group.get('question', '')

        # Retrieve relevant documentation
        retrieved_docs = retriever.retrieve(question_text, top_k=3)
        if not retrieved_docs:
            retrieved_docs = ["No documentation retrieved"]

        # Load test case
        test_case = load_test_case_from_file(
            test_case_path=config["test_case_path"],
            task_name=task_name,
            is_directed=is_directed,
            is_weighted=is_weighted
        )

        # Sample args
        sample_args = graph_instances[0]['args'] if graph_instances else None

        # Create prompt
        prompt = create_code_generation_prompt(
            question_text=question_text,
            is_weighted=is_weighted,
            is_directed=is_directed,
            args=sample_args,
            retrieved_docs=retrieved_docs,
            retrieval_method="Sentence-BERT",
            test_case=test_case
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
            'retrieved_docs_count': len(retrieved_docs),
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
                    'error': 'Graph not found'
                })
                continue

            edge_list = graphs_data[graph_id]['graph']

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
                'correct_rate': correct_rate
            }
            print(f"\n✓ Task complete: {correct_count}/{total} correct ({correct_rate*100:.1f}%)")

        all_results.append(task_results)

        # Save task results
        output_file = output_dir / f"{task_name}_results.json"
        save_dict_to_json(task_results, str(output_file))

    # Save combined results
    combined_output = output_dir / "all_results.json"
    save_dict_to_json({'results': all_results, 'model': model_name, 'dataset': dataset_version},
                      str(combined_output))
    print(f"\n✓ Results saved to: {combined_output}")

    # Evaluate
    evaluation_results = evaluate_all_results(all_results)
    eval_output = output_dir / "evaluation_metrics.json"
    save_dict_to_json(evaluation_results, str(eval_output))
    print(f"✓ Evaluation metrics saved to: {eval_output}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sentence-BERT + Coding Agent (unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--benchmark", type=str, required=True, choices=["complexgraph", "gtools"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="small")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_correction_rounds", type=int, default=3)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--tasks", type=str, nargs='+', default=None)

    args = parser.parse_args()
    config = get_dataset_config(args.benchmark, args.dataset)

    start_time = time.time()
    run_sentbert_coding_experiment(
        model_name=args.model,
        dataset_version=args.dataset,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_correction_rounds=args.max_correction_rounds,
        max_instances=args.max_instances,
        task_filter=args.tasks
    )
    end_time = time.time()
    print(f"Total elapsed time: {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
