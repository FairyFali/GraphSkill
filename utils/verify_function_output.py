import ast
import math
from collections.abc import Mapping, Sequence

SPECIAL_SET_EQ = {
    "highest_clustered_node_in_shortest_path",
    "max_clique",
    "clustering_and_shortest_path"
}

SPECIAL_MULTiset_EQ = {
    # add tasks where order doesn't matter but multiplicity does
    # e.g., if ever needed: "some_task"
}

SPECIAL_ORDERLESS_LIST_EQ = {
    # add tasks where list order doesn't matter
}

FLOAT_ABS_TOL = 1e-9
FLOAT_REL_TOL = 1e-9


def _maybe_literal_eval(x):
    """If x is a string that looks like a Python literal, parse it safely."""
    if isinstance(x, str):
        s = x.strip()
        # quick filter: only attempt eval if it resembles a literal
        if s and (s[0] in "[{('\"-0123456789" or s in ("True", "False", "None")):
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


def _coerce_numeric_strings(x):
    """Convert numeric strings to int/float when safe."""
    if isinstance(x, str):
        s = x.strip()
        # int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # float
        try:
            # avoid turning things like "1e3" not caught above? allow it:
            v = float(s)
            # but don't convert strings that are clearly non-numeric like "nan" unless you want to
            return v
        except Exception:
            return x
    return x


def _normalize(obj):
    """
    Canonicalize objects so equivalent values compare equal across types.
    - Parses string-literals when possible
    - Converts numeric strings
    - Converts tuples -> lists
    - Normalizes nested structures recursively
    """
    obj = _maybe_literal_eval(obj)
    obj = _coerce_numeric_strings(obj)

    # Normalize NaN (NaN != NaN by default)
    if isinstance(obj, float) and math.isnan(obj):
        return ("__NaN__",)

    # Mappings (dict-like)
    if isinstance(obj, Mapping):
        # normalize keys and values; sort keys via string repr for stability
        items = [(_normalize(k), _normalize(v)) for k, v in obj.items()]
        items.sort(key=lambda kv: repr(kv[0]))
        return ("__dict__", tuple(items))

    # Sequences (list/tuple), but not strings/bytes
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [ _normalize(x) for x in obj ]

    return obj


def _float_equal(a, b, abs_tol=FLOAT_ABS_TOL, rel_tol=FLOAT_REL_TOL):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # handle NaN canonical form already
        return math.isclose(float(a), float(b), abs_tol=abs_tol, rel_tol=rel_tol)
    return False


def _deep_equal(a, b, abs_tol=FLOAT_ABS_TOL, rel_tol=FLOAT_REL_TOL):
    """
    Deep equality with float tolerance for numeric leaves.
    Assumes inputs already normalized by _normalize.
    """
    # numeric tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _float_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)

    # exact match for primitives
    if type(a) == type(b) and a == b:
        return True

    # list deep compare
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y, abs_tol=abs_tol, rel_tol=rel_tol) for x, y in zip(a, b))

    # normalized dict representation
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) > 0 and a[0] == "__dict__" and b[0] == "__dict__":
        return a == b

    return a == b


def verify_function_output(function_name, input_data, expected_output,
                          abs_tol=FLOAT_ABS_TOL, rel_tol=FLOAT_REL_TOL):
    """
    Robust evaluator:
    - normalizes stringified literals and numeric strings
    - supports float tolerance
    - supports special-case set equality for tasks like max_clique
    """
    a = _normalize(input_data)
    b = _normalize(expected_output)

    # Special-case: order-insensitive *set* equality (duplicates ignored)
    if function_name in SPECIAL_SET_EQ:
        try:
            return set(a) == set(b)
        except TypeError:
            # unhashable elements; fallback to deep equality
            return _deep_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)

    # Special-case: order-insensitive but multiplicity-sensitive
    if function_name in SPECIAL_MULTiset_EQ:
        from collections import Counter
        try:
            return Counter(a) == Counter(b)
        except TypeError:
            return _deep_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)

    # Special-case: list order doesn't matter (but elements hashable)
    if function_name in SPECIAL_ORDERLESS_LIST_EQ:
        try:
            return sorted(a) == sorted(b)
        except Exception:
            return _deep_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)

    # Default: deep structural equality with float tolerance
    return _deep_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol)

    