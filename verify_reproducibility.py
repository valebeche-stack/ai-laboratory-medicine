"""Check exact repeatability in two fresh processes in the current environment."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    script = Path(__file__).resolve().parent / "ai_laboratory_demo.py"
    with tempfile.TemporaryDirectory() as directory:
        records = []
        for run in (1, 2):
            output = Path(directory) / f"run_{run}.json"
            subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                cwd=directory, check=True,
            )
            records.append(json.loads(output.read_text(encoding="utf-8")))
        if records[0] != records[1]:
            differences = [key for key in records[0] if records[0][key] != records[1][key]]
            raise SystemExit(f"FAIL: outputs differ in {differences}")
    print("PASS: split, predictions, probabilities, coefficients, metrics and metadata match exactly.")
    print("This checks computational repeatability only, not clinical validity or utility.")


if __name__ == "__main__":
    main()
