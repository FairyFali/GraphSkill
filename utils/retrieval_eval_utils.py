"""
Retrieval evaluation utilities for measuring how well RAG retrieval
captures the required NetworkX functions for each graph task.

Provides:
- evaluate_retrieval_correctness(): per-task recall/precision/F1
- aggregate_retrieval_metrics(): micro/macro metrics across tasks
"""

import re
from typing import Dict, List, Optional

import numpy as np
import json


# Mapping of composite tasks to their required NetworkX functions.
# For standard (non-composite) tasks, the task name itself is the required function.
TASK_NX_FUNC = {'clustering_and_shortest_path': ['clustering', 'shortest_path_length'], 'highest_clustered_node_in_shortest_path': ['clustering', 'all_shortest_paths'], 'largest_component_and_diameter': ['is_connected', 'diameter', 'connected_components'], 'shortest_path_and_eular_tour': ['has_eulerian_path', 'diameter', 'clustering'], 'pair_tightness_score': ['shortest_path_length', 'common_neighbors', 'clustering'], 'flow_aware_local_clustering_among_cc': ['connected_components', 'maximum_flow', 'clustering'], 'endpoint_aware_flow_score': ['maximum_flow', 'clustering'], 'scc_and_diameter_nested': ['strongly_connected_components', 'diameter'], 'scc_and_eulerian_feasibility': ['has_eulerian_path', 'strongly_connected_components'], 'strongly_connected_components': ['strongly_connected_components'], 'maximum_flow': ['maximum_flow'], 'is_bipartite': ['is_bipartite'], 'find_cycle': ['find_cycle'], 'shortest_path_length': ['shortest_path_length'], 'is_regular': ['is_regular'], 'enumerate_all_cliques': ['enumerate_all_cliques'], 'has_node': ['has_node'], 'is_distance_regular': ['is_distance_regular'], 'has_eulerian_path': ['has_eulerian_path'], 'degree': ['degree'], 'number_of_edges': ['number_of_edges'], 'has_edge': ['has_edge'], 'number_of_nodes': ['number_of_nodes'], 'clustering': ['clustering'], 'is_connected': ['is_connected'], 'connected_components': ['connected_components'], 'diameter': ['diameter'], 'maximum_independent_set': ['maximum_independent_set'], 'topological_sort': ['topological_sort'], 'has_path': ['has_path'], 'max_clique': ['max_clique'], 'common_neighbors': ['common_neighbors']}
retrieval_groundtruth_path = "data/retrieval_groundtruth.json"

with open(retrieval_groundtruth_path, 'r', encoding='utf-8') as f:
    RETRIEVAL_GROUNDTRUTH = json.load(f)

def evaluate_retrieval_correctness(
    retrieved_docs: List[str],
    task_name: str,
) -> dict:
    """
    Evaluate whether retrieved docs contain the required NetworkX functions.

    A function is a HIT if its name appears (case-insensitive) in any
    retrieved document.

    Args:
        retrieved_docs: List of retrieved documentation strings.
        task_name: Name of the graph task being evaluated.

    Returns:
        Dict with:
          - per_function: {func_name: bool} hit/miss per required function
          - summary: dict with required, hit, hit_doc, all_hit,
                     predicted, recall, precision, f1
    """
    assert retrieved_docs is not None, "Retrieved docs must be provided for evaluation."
    required_docs = RETRIEVAL_GROUNDTRUTH.get(task_name, [])
    assert required_docs is not None, f"No groundtruth for task: {task_name}"

    num_hits = len(set(required_docs) & set(retrieved_docs))
    num_retrieved = len(retrieved_docs)
    num_required = len(required_docs)

    recall = (num_hits / num_required) if num_required else 1.0
    precision = (num_hits / num_retrieved) if num_retrieved else 0.0
    f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
    summary = {
        "required": num_required,
        "predicted": num_retrieved,
        "hit": num_hits,
        "all_hit": (num_hits == num_required) if num_required else True,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    return {"summary": summary}


def aggregate_retrieval_metrics(all_results: List[dict]) -> dict:
    """
    Aggregate per-task retrieval evaluation into micro/macro metrics.

    Expects each element of *all_results* to optionally contain a
    ``retrieval_evaluation`` key (as produced by
    ``evaluate_retrieval_correctness``).

    Args:
        all_results: List of task-result dicts, each with an optional
            ``retrieval_evaluation`` entry containing a ``summary`` dict.

    Returns:
        Dict with aggregated metrics (empty dict if no tasks have
        retrieval evaluation). Keys include micro_recall,
        micro_precision, micro_f1, macro_recall, macro_precision,
        macro_f1, tasks_with_all_hit, all_hit_rate, etc.
    """
    total_required = 0
    total_hits = 0
    total_predicted = 0
    total_all_hit = 0
    num_tasks_with_eval = 0
    recall_list = []
    precision_list = []
    f1_list = []

    for task_result in all_results:
        ret_eval = task_result.get('retrieval_evaluation')
        if ret_eval and 'summary' in ret_eval:
            s = ret_eval['summary']
            total_required += s['required']
            total_predicted += s['predicted']
            total_hits += s['hit']
            total_all_hit += int(s['all_hit'])
            recall_list.append(s['recall'])
            precision_list.append(s['precision'])
            f1_list.append(s['f1'])
            num_tasks_with_eval += 1

    if num_tasks_with_eval == 0:
        return {}

    micro_recall = total_hits / total_required if total_required > 0 else 0.0
    micro_precision = total_hits / total_predicted if total_predicted > 0 else 0.0
    micro_f1 = (2 * micro_recall * micro_precision) / (micro_recall + micro_precision) if (micro_recall + micro_precision) > 0 else 0.0
    macro_recall = float(np.mean(recall_list)) if recall_list else 0.0
    macro_precision = float(np.mean(precision_list)) if precision_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        'num_tasks_evaluated': num_tasks_with_eval,
        'total_required': total_required,
        'total_retrieved': total_predicted,
        'total_hits': total_hits,
        'micro_recall': micro_recall,
        'micro_precision': micro_precision,
        'micro_f1': micro_f1,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_f1': macro_f1,
        'tasks_with_all_hit': total_all_hit,
        'all_hit_rate': total_all_hit / num_tasks_with_eval,
    }


def print_retrieval_summary(retrieval_summary: dict) -> None:
    """Print a formatted retrieval evaluation summary to stdout."""
    if not retrieval_summary:
        return
    print(f"\n{'='*70}")
    print(f"Retrieval Evaluation Summary")
    print(f"{'='*70}")
    print(f"  Tasks evaluated: {retrieval_summary['num_tasks_evaluated']}")
    print(f"  Total required docs: {retrieval_summary['total_required']}")
    print(f"  Total retrieved docs: {retrieval_summary['total_retrieved']}")
    print(f"  Total hits: {retrieval_summary['total_hits']}")
    print(f"  Micro recall: {retrieval_summary['micro_recall']:.2%}")
    print(f"  Micro precision: {retrieval_summary['micro_precision']:.2%}")
    print(f"  Micro F1: {retrieval_summary['micro_f1']:.2%}")
    print(f"  Macro recall: {retrieval_summary['macro_recall']:.2%}")
    print(f"  Macro precision: {retrieval_summary['macro_precision']:.2%}")
    print(f"  Macro F1: {retrieval_summary['macro_f1']:.2%}")
    print(f"  Tasks with all doc hit: {retrieval_summary['tasks_with_all_hit']}/{retrieval_summary['num_tasks_evaluated']} ({retrieval_summary['all_hit_rate']:.2%})")
    print(f"{'='*70}\n")
