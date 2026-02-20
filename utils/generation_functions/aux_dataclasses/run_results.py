from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

MIN_CASES = 8
@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float



