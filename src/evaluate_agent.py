
"""Part D: Isolated evaluation of the refund agent."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from src import refund_agent


def evaluate_agent():
    original_out = refund_agent.OUT
    original_state = refund_agent.STATE

    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)

        refund_agent.OUT = temp / "outputs"
        refund_agent.STATE = temp / "state"

        refund_agent.OUT.mkdir(parents=True, exist_ok=True)
        refund_agent.STATE.mkdir(parents=True, exist_ok=True)

        try:
            for _, row in scenarios.iterrows():
                scenario_id = row["scenario_id"]
                message = row["message"]
                expected_status = row["expected_outcome"]
                expected_refund = float(row["expected_refund_inr"])

                if expected_status == "FALLBACK":
                    expected_status = "FALLBACK_MAX_ITERATIONS"

                result = refund_agent.run_agent(
                    user_message=message,
                    scenario_id=scenario_id
                )

                actual_status = result["status"]
                actual_refund = 0.0

                # Read refund amounts from tool observations.
                for step in result["steps"]:
                    tool_name = step.get("tool", step.get("name"))
                    observation = step.get("observation", {})

                    if isinstance(observation, str):
                        observation = json.loads(observation)

                    if (
                        tool_name in (
                            "calculate_refund",
                            "issue_refund",
                            "escalate_to_human"
                        )
                        and isinstance(observation, dict)
                        and "refund_inr" in observation
                    ):
                        actual_refund = float(observation["refund_inr"])

                status_passed = actual_status == expected_status
                amount_passed = (
                    abs(actual_refund - expected_refund) <= 0.01
                )

                record = {
                    "scenario_id": scenario_id,
                    "user_message": message,
                    "expected_status": expected_status,
                    "status": actual_status,
                    "actual_status": actual_status,
                    "expected_refund_inr": expected_refund,
                    "actual_refund_inr": actual_refund,
                    "status_passed": status_passed,
                    "amount_passed": amount_passed,
                    "passed": status_passed and amount_passed,
                    "final_answer": result["final_answer"],
                    "steps": result["steps"],
                    "llm_calls": result["llm_calls"],
                    "tokens_used": result["tokens_used"]
                }

                results.append(record)

                print(
                    f"{scenario_id}: {actual_status} | "
                    f"Refund: {actual_refund:.2f} | "
                    f"Passed: {record['passed']} | "
                    f"Steps: {len(result['steps'])}"
                )

        finally:
            refund_agent.OUT = original_out
            refund_agent.STATE = original_state

    output_path = original_out / "agent_runs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in results:
            file.write(json.dumps(record, default=str) + "\n")

    passed = sum(record["passed"] for record in results)

    print(f"\nRESULT: {passed}/{len(results)} scenarios passed")
    print("Saved to:", output_path)


if __name__ == "__main__":
    evaluate_agent()
