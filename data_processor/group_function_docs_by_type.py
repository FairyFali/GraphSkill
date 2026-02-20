import json
from collections import defaultdict

def group_by_third_component(records):
    """
    Group a list of NetworkX metadata dicts by the third word in `qualname`.

    Parameters
    ----------
    records : list[dict]
        Each dict must have keys 'function' and 'qualname'.

    Returns
    -------
    dict[str, dict]
        {third_word: {function_name: record_dict, ...}, ...}
    """
    grouped = defaultdict(dict)

    for rec in records:
        parts = rec["qualname"].split(".")
        if len(parts) < 3:
            raise ValueError(f"Qualname {rec['qualname']} has fewer than 3 components")
        third_word = parts[2]
        grouped[third_word][rec["function"]] = rec

    return dict(grouped)   # cast back to a plain dict if desired


if __name__ == "__main__":
    with open("../evaluation_dataset/GWild_docs/function_pages/task_examples.json") as f:
        data = json.load(f)
    grouped_dict = group_by_third_component(data)
    with open("../evaluation_dataset/GWild_docs/function_pages/grouped_task_examples.json", "w") as f:
        json.dump(grouped_dict, f)
