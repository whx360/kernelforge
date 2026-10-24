"""Project entry point; no installation or global environment changes required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from kernelforge.cli import main
    raise SystemExit(main(ROOT))
