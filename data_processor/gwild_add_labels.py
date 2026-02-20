import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
import json
from tqdm import tqdm

# Regex: fenced block starting with ```python and ending with ```
CODEBLOCK_RE = re.compile(r"```python\s*([\s\S]*?)\s*```", re.IGNORECASE)

def execute(code: str) -> str:
    """Execute *code* and return everything it printed to stdout."""
    buf = io.StringIO()
    # Separate namespace for each block
    namespace = {}
    try:
        with redirect_stdout(buf):
            exec(code, namespace)
    except Exception as exc:
        # Forward traceback to caller’s stdout capture
        import traceback
        traceback.print_exc(file=buf)
    return buf.getvalue()
def main(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    updated = data
    for i, datum in tqdm(enumerate(data)):
        code = CODEBLOCK_RE.findall(datum["output"])[0]
        captured = execute(code)
        label = captured if captured else None
        updated[i]["label"] = label
    return updated

output_path = "/data/chenglin/GraphTutor/Dataset/training_dataset/GWild_labelled.json"
json_path = "/data/chenglin/GraphTutor/Dataset/training_dataset/GWild.json"
updated = main(json_path)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False)

