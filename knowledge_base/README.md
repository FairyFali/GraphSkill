# Knowledge Base

This directory contains documentation retrieval and prompt generation utilities.

## Structure

### doc_retrieval/
- `retrieve_nx_documentation_llm.py` - Retrieve NetworkX documentation using LLM
  - Function: `retrieve_doc(task_description, method, top_k)`
- `read_RAG_document.py` - Read and parse RAG documentation files

### prompt_generation/
- `nlgraph_prompts_generation_algorithm.py` - Generate prompts for NLGraph tasks
  - Algorithm-focused prompt templates

### paraphrasing/
- `paraphrase_task_descriptions.py` - Paraphrase task descriptions for data augmentation
  - Used to increase training data diversity

## Usage Examples

### Retrieve NetworkX documentation:
```python
from knowledge_base.doc_retrieval.retrieve_nx_documentation_llm import retrieve_doc

docs = retrieve_doc(
    task_description="Find connected components",
    method="bm25",  # or "sentence_bert", "tf_idf"
    top_k=5
)
```

### Read RAG documents:
```python
from knowledge_base.doc_retrieval.read_RAG_document import read_document

doc_content = read_document(doc_path="crawl_documentation/nx_docs.json")
```

### Paraphrase task descriptions:
```python
from knowledge_base.paraphrasing.paraphrase_task_descriptions import paraphrase

variants = paraphrase(
    original_description="Calculate the shortest path between two nodes",
    num_variants=3
)
```
