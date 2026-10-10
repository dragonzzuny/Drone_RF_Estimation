"""Bindings to frozen native-RF model/data; no legacy source edits."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/source_interaction_20261010'))
import train_comparison as worker
w = worker.watch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

phase = load('consistent_original_phase', ROOT / 'experiments/rfuav_phase_average_20261009/phase_core.py')
eight = load('consistent_original_eight', ROOT / 'experiments/phase_eight_20261011/core.py')
OUT = ROOT / 'local/phase_consistent_20261011_v1'
PUBLIC = ROOT / 'reports/2026-10-11'
PARENT = ROOT / 'local/native_frequency_20261009_v1/gpu/SELECTED_005.pt'
PARENT_SHA = '19888f8823fbb6ade6333d40bd849dec3202e522ded016e7265eaced6b52aa49'
PREPARATION = ROOT / 'local/native_frequency_20261009_v1/preparation'

