import sys
from pathlib import Path

SCENARIOS = ["animals", "cars", "ecomm", "medical", "mmqa", "movie"]

def main():
    root = Path(__file__).resolve().parent.parent
    sys.path.append(str(root / "SemBench" / "src"))

    from scenario.movie.runner.palimpzest_runner.palimpzest_runner import PalimpzestRunner

    for scenario in SCENARIOS:
        print(f"\nRunning SemBench scenario: {scenario}")

        runner = PalimpzestRunner(
            use_case=scenario,
            scale_factor=1
        )

        runner.run_all_queries()

if __name__ == "__main__":
    main()
