import tempfile
import unittest
from pathlib import Path

from app.agent.evals import load_agent_eval_cases, run_agent_eval_suite


REAL_DATASET_PATH = Path("data/agent_eval_cases.json")


class AgentEvalRunnerTestCase(unittest.TestCase):
    def test_load_agent_eval_cases_reads_external_dataset(self) -> None:
        cases = load_agent_eval_cases(REAL_DATASET_PATH)

        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(cases[0].id, "direct_answer_001")
        self.assertEqual(cases[-1].id, "budget_timeout_001")

    def test_run_agent_eval_suite_passes_real_dataset_and_writes_report(self) -> None:
        cases = load_agent_eval_cases(REAL_DATASET_PATH)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "agent_eval_report.json"
            report = run_agent_eval_suite(
                cases,
                dataset_path=REAL_DATASET_PATH,
                output_path=output_path,
            )

            self.assertEqual(report.total_cases, len(cases))
            self.assertEqual(report.failed_cases, 0)
            self.assertFalse(report.threshold_failures)
            self.assertEqual(report.metrics["route_accuracy"], 1.0)
            self.assertEqual(report.metrics["tool_selection_accuracy"], 1.0)
            self.assertEqual(report.metrics["task_success_rate"], 1.0)
            self.assertTrue(output_path.exists())

    def test_real_dataset_covers_cache_memory_and_budget_behaviors(self) -> None:
        cases = load_agent_eval_cases(REAL_DATASET_PATH)
        report = run_agent_eval_suite(cases, dataset_path=REAL_DATASET_PATH, output_path=None)
        indexed_results = {result.case_id: result for result in report.cases}

        self.assertTrue(indexed_results["cache_hit_001"].observed.cached)
        self.assertTrue(indexed_results["memory_followup_001"].observed.memory_applied)
        self.assertEqual(
            indexed_results["budget_timeout_001"].observed.refusal_reason,
            "workflow_timeout_exceeded",
        )
        self.assertEqual(
            indexed_results["policy_refusal_001"].observed.tool_names,
            ["get_session_memory", "get_cached_answer"],
        )

    def test_regression_gate_fails_when_expectation_is_broken(self) -> None:
        cases = load_agent_eval_cases(REAL_DATASET_PATH)
        broken_payload = cases[0].model_dump(mode="python")
        broken_payload["expected"]["route"] = "retrieve_vector"
        broken_case = type(cases[0]).model_validate(broken_payload)

        report = run_agent_eval_suite([broken_case], dataset_path="broken-case", output_path=None)

        self.assertEqual(report.failed_cases, 1)
        self.assertTrue(report.threshold_failures)
        self.assertEqual(report.metrics["route_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
