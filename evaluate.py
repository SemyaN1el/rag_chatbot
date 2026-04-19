import argparse
import sys
from pathlib import Path

from app.agent.evals import (
    DEFAULT_AGENT_EVAL_DATASET_PATH,
    DEFAULT_AGENT_EVAL_REPORT_PATH,
    format_agent_eval_report,
    load_agent_eval_cases,
    run_agent_eval_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка поведения agent runtime через regression suite."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        help=f"Путь к JSON-файлу с agent eval cases. По умолчанию: {DEFAULT_AGENT_EVAL_DATASET_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AGENT_EVAL_REPORT_PATH,
        help=f"Куда сохранить JSON-отчёт. По умолчанию: {DEFAULT_AGENT_EVAL_REPORT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or DEFAULT_AGENT_EVAL_DATASET_PATH
    cases = load_agent_eval_cases(dataset_path)
    report = run_agent_eval_suite(
        cases,
        dataset_path=dataset_path,
        output_path=args.output,
    )

    print(format_agent_eval_report(report))
    print()
    print(f"JSON-отчёт сохранён в: {args.output}")

    if report.failed_cases or report.threshold_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
