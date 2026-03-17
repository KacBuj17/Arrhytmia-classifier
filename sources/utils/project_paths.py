from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def from_root(rel_path):
    return PROJECT_ROOT.joinpath(*Path(rel_path).parts).resolve()
