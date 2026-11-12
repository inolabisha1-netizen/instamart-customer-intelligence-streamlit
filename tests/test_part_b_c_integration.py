
"""Verify Part B customer segments are used correctly in Part C routing."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
tickets = pd.read_csv(OUTPUTS / "triage_results.csv", keep_default_na=False)

# Validate required columns and customer identifiers.
assert {"customer_id", "segment_name", "is_premium"}.issubset(segments.columns)
assert {
    "ticket_id", "customer_id", "status", "category",
    "segment_name", "priority", "route", "tokens_used"
}.issubset(tickets.columns)

assert len(tickets) == 164
assert tickets["ticket_id"].is_unique
assert segments["customer_id"].is_unique

# Identify Premium customers from the actual Part B output.
premium = set(
    segments.loc[
        segments["is_premium"].astype(str).str.lower() == "true",
        "customer_id"
    ]
)

segment_names = segments.set_index("customer_id")["segment_name"].to_dict()

human_statuses = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
}

valid_statuses = {"classified_json", "classified_keyword"}

for _, ticket in tickets.iterrows():
    customer_id = ticket["customer_id"]
    status = ticket["status"]
    category = ticket["category"]

    # Verify the segment label came from Part B.
    expected_segment = segment_names.get(customer_id, "")
    assert ticket["segment_name"] == expected_segment, (
        f"Segment mismatch: {ticket['ticket_id']}"
    )

    # Independently calculate the expected route and priority.
    if status in human_statuses:
        expected_route = "human_review"
        expected_priority = "NORMAL"

    elif status in valid_statuses and category == "fraud":
        expected_route = "security_team"
        expected_priority = "HIGH"

    elif status in valid_statuses and customer_id in premium:
        expected_route = "premium_desk"
        expected_priority = "HIGH"

    elif status in valid_statuses:
        expected_route = "support_queue"
        expected_priority = "NORMAL"

    else:
        raise AssertionError(
            f"Unexpected status: {ticket['ticket_id']} — {status}"
        )

    assert ticket["route"] == expected_route, (
        f"Incorrect route: {ticket['ticket_id']}"
    )

    assert ticket["priority"] == expected_priority, (
        f"Incorrect priority: {ticket['ticket_id']}"
    )

print("PART B → PART C INTEGRATION PASSED")
print("Tickets verified:", len(tickets))
print("Premium customers in Part B:", len(premium))
print("Premium desk tickets:", (tickets["route"] == "premium_desk").sum())
print("Fraud tickets routed to security:", (tickets["route"] == "security_team").sum())
print("Human review tickets:", (tickets["route"] == "human_review").sum())
