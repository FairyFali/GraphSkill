from pathlib import Path
import json
import os
import re
import pathlib


# ----------------- Load GT function signatures -----------------
with open("../../evaluation_dataset/GWild_docs/function_pages/task_examples.json", "r", encoding="utf-8") as f:
    nx_examples = json.load(f)


# ----------------- Task config (must include nx_func) -----------------
all_tasks = {
    "unweighted_undirected_tasks": {
        "tasks": [
            "clustering_and_shortest_path",
            "highest_clustered_node_in_shortest_path",
            "largest_component_and_diameter",
            "shortest_path_and_eular_tour",
        ],
        "nx_func": {
            "clustering_and_shortest_path": ["clustering", "shortest_path_length"],
            "highest_clustered_node_in_shortest_path": ["clustering", "all_shortest_paths"],
            "largest_component_and_diameter": ["is_connected", "diameter", "connected_components"],
            "shortest_path_and_eular_tour": ["has_eulerian_path", "diameter", "clustering"],
        },
        "directed": False,
        "weighted": False,
    },
    "weighted_undirected_tasks": {
        "tasks": ["pair_tightness_score", "flow_aware_local_clustering_among_cc"],
        "nx_func": {
            "pair_tightness_score": ["shortest_path_length", "common_neighbors", "clustering"],
            "flow_aware_local_clustering_among_cc": ["connected_components", "maximum_flow", "clustering"],
        },
        "directed": False,
        "weighted": True,
    },
    "weighted_directed_tasks": {
        "tasks": ["endpoint_aware_flow_score"],
        "nx_func": {"endpoint_aware_flow_score": ["maximum_flow", "clustering"]},
        "directed": True,
        "weighted": True,
    },
    "unweighted_directed_tasks": {
        "tasks": ["scc_and_diameter_nested", "scc_and_eulerian_feasibility"],
        "nx_func": {
            "scc_and_diameter_nested": ["strongly_connected_components", "diameter"],
            "scc_and_eulerian_feasibility": ["has_eulerian_path", "strongly_connected_components"],
        },
        "directed": True,
        "weighted": False,
    },
}


# ----------------- Paths -----------------
def get_base_retrieved_docs_dir(retrieval_method: str) -> pathlib.Path:
    # Same convention you used
    if retrieval_method == "agentic_RAG":
        return Path("../../LLM_generation_results/gcoder_tasks/agentic_RAG_llm_descision_retrieved_docs/edge_list")
    return Path(f"../../LLM_generation_results/gcoder_tasks/{retrieval_method}_retrieved_docs/edge_list")

def get_output_dir(retrieval_method: str) -> pathlib.Path:
    return Path(f"retrieval_correctness/{retrieval_method}")


# ----------------- Helpers -----------------
def find_json_files(base_folder: Path):
    out = []
    for root, _, files in os.walk(base_folder):
        for fn in files:
            if fn.endswith(".json"):
                out.append(Path(root) / fn)
    return out

def get_nx_signature(nx_docs, func_name: str) -> str:
    for d in nx_docs:
        if d.get("function") == func_name:
            return d.get("signature", "")
    return ""

def normalize_retrieved_text(retrieved_docs) -> str:
    """
    retrieved_docs could be str, list[str], dict, list[dict], etc.
    We stringify everything robustly.
    """
    if retrieved_docs is None:
        return ""
    if isinstance(retrieved_docs, str):
        return retrieved_docs
    if isinstance(retrieved_docs, list):
        return "\n".join(normalize_retrieved_text(x) for x in retrieved_docs)
    if isinstance(retrieved_docs, dict):
        # common fields to include if present
        parts = []
        for k in ["function", "signature", "docstring", "text", "content", "description", "summary"]:
            if k in retrieved_docs and isinstance(retrieved_docs[k], str):
                parts.append(retrieved_docs[k])
        if parts:
            return "\n".join(parts)
        return json.dumps(retrieved_docs, ensure_ascii=False)
    return str(retrieved_docs)

def evaluate_retrieval_correctness(retrieved_docs, required_funcs, nx_examples) -> dict:
    """
    A required function is a HIT if:
      - function name appears in retrieved text OR
      - GT signature appears in retrieved text (if available)
    """
    text = normalize_retrieved_text(retrieved_docs)

    per_fn = {}
    hits = 0
    for fn in required_funcs:
        sig = get_nx_signature(nx_examples, fn)

        patterns = []
        if fn:
            patterns.append(re.escape(fn))
        if sig:
            patterns.append(re.escape(sig))

        hit = any(re.search(p, text) for p in patterns)
        per_fn[fn] = {"hit": bool(hit), "signature": sig}
        hits += int(bool(hit))

    summary = {
        "required": len(required_funcs),
        "hit": hits,
        "recall": (hits / len(required_funcs)) if required_funcs else 1.0,
        "all_hit": (hits == len(required_funcs)) if required_funcs else True,
        "predicted": len(retrieved_docs) if retrieved_docs else 0,
        "precision": (hits / len(retrieved_docs)) if retrieved_docs else 1.0,
    }
    return {"per_function": per_fn, "summary": summary}


# ----------------- Main -----------------
def main(retrieval_methods):
    for retrieval_method in retrieval_methods:
        base_dir = get_base_retrieved_docs_dir(retrieval_method)
        out_dir = get_output_dir(retrieval_method)
        out_dir.mkdir(parents=True, exist_ok=True)

        full_report = {}

        for graph_type, config in all_tasks.items():
            if graph_type == "weighted_undirected_tasks":
                is_directed, is_weighted = "undirected", "weighted"
            elif graph_type == "unweighted_directed_tasks":
                is_directed, is_weighted = "directed", "unweighted"
            elif graph_type == "weighted_directed_tasks":
                is_directed, is_weighted = "directed", "weighted"
            elif graph_type == "unweighted_undirected_tasks":
                is_directed, is_weighted = "undirected", "unweighted"
            else:
                raise ValueError(f"Invalid graph type: {graph_type}")

            retrieve_root = base_dir / is_directed / is_weighted
            json_files = find_json_files(retrieve_root)
            if not json_files:
                full_report[graph_type] = {"error": f"No retrieved-doc JSON found under {retrieve_root.as_posix()}"}
                continue

            # You previously used [0]; keep same behavior deterministically:
            retrieved_docs_path = sorted(json_files)[0]

            with open(retrieved_docs_path, "r", encoding="utf-8") as f:
                retrieved_by_task = json.load(f)

            graph_report = {
                "retrieved_docs_path": retrieved_docs_path.as_posix(),
                "tasks": {}
            }

            for task_name in config["tasks"]:
                required_funcs = config["nx_func"].get(task_name, [])
                retrieved_docs = retrieved_by_task.get(task_name)
                graph_report["tasks"][task_name] = evaluate_retrieval_correctness(
                    retrieved_docs=retrieved_docs,
                    required_funcs=required_funcs,
                    nx_examples=nx_examples,
                )

            full_report[graph_type] = graph_report

        out_path = out_dir / f"{retrieval_method}_retrieval_eval.json"
        out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    retrieval_methods = ["tf_idf", "sentence_bert", "graph_team", "agentic_RAG", "flat_agent"]  # "graph_tool",  finished: "agentic_RAG" add: "tf_idf", "sentence_bert", "graph_team", ...
    main(retrieval_methods)