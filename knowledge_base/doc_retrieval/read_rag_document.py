import json
from langchain_core.documents import Document
from typing import List, Dict


def chunk_code_entries(code_library: List[Dict[str, str]]) -> List[Document]:
    """
    Converts each entry into a single Document object for indexing.
    """
    documents = []
    for entry in code_library:
        content = entry["code_snippet"]
        metadata = {"task_name": entry["task_name"]}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def load_code_library(json_path: str) -> List[Dict[str, str]]:
    """
    Loads the code library from a JSON file.

    The file is expected to contain a top-level JSON object
    with each key corresponding to a 'task_name'
    and its value being the 'code_snippet' (text).
    For example:
    {
        "Graph_Views": "...long text here...",
        "GEXF": "...another long text here..."
    }

    This function returns a list of dictionaries, each with:
      {
          "task_name": <the JSON key>,
          "code_snippet": <the JSON value>
      }
    """
    with open(json_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)  # data is now a dict: {key -> text, key -> text, ...}

    code_library = []
    for key, snippet_text in data.items():
        code_library.append({
            "task_name": key,
            "code_snippet": snippet_text
        })
    
    return code_library

def chunk_code_entries(code_library: List[Dict[str, str]]) -> List[Document]:
    """
    Converts each entry into a single Document object for indexing.
    """
    documents = []
    for entry in code_library:
        content = entry["code_snippet"]
        metadata = {"task_name": entry["task_name"]}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents
