"""Lightweight adapter for saved toy results; deliberately has no torch import."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['QWENCH_REPRO_OUT']).resolve()
P = json.loads((ROOT / 'configs/toy-scalar-comparison-v1.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
