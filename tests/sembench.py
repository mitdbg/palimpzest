import sys
import importlib
from pathlib import Path

SCENARIOS = ["movie", "animals"] #"cars", "ecomm", "medical", "mmqa"

def main():
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "SemBench" / "src"))

    for scenario in SCENARIOS:
        print(f"\nRunning SemBench scenario: {scenario}")

        module = importlib.import_module(
            f"scenario.{scenario}.runner.palimpzest_runner.palimpzest_runner"
        )
        PalimpzestRunner = module.PalimpzestRunner

        runner = PalimpzestRunner(
            use_case=scenario,
            scale_factor=1
        )

        runner.run_all_queries()

if __name__ == "__main__":
    main()
