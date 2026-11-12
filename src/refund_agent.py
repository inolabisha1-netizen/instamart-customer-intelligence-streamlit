"""Part D: database-backed refund tools and bounded ReAct loop."""
from pathlib import Path
import json,sqlite3,time,uuid
from datetime import datetime,timezone
import mock_llm
from src.policy_rag import PolicyRAG
ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/"data/instamart.db"; OUT=ROOT/"outputs"; STATE=ROOT/"state"
rag=PolicyRAG()
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row=con.execute("select order_id,status,order_amount,delivery_minutes from orders where order_id=?",(order_id,)).fetchone()
    return {"error":"order not found"} if not row else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
def search_policy(query): return {"results":rag.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0
    return {"refund_inr":round(amount,2)}
def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(exist_ok=True)
    path = OUT / "refund_ledger.jsonl"

    if tool == "issue_refund" and path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)

                if (
                    record.get("tool") == "issue_refund"
                    and record.get("order_id") == order_id
                ):
                    return {
                        "error": "refund already issued",
                        "order_id": order_id,
                        "refund_inr": record.get("refund_inr")
                    }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"refund_inr": refund_inr}
def issue_refund(order_id, refund_inr):
    order = get_order(order_id)

    if "error" in order:
        return {"error": "order not found"}

    if order["status"] != "completed":
        return {"error": "refund denied: order is not completed"}

    if order["delivery_minutes"] is None or float(order["delivery_minutes"]) <= 45:
        return {"error": "refund denied: delivery time is not eligible"}

    eligible_amount = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )["refund_inr"]

    requested_amount = round(float(refund_inr), 2)

    if requested_amount <= 0 or abs(requested_amount - eligible_amount) > 0.01:
        return {"error": "refund amount does not match eligible amount"}

    if requested_amount > 1000:
        return {"error": "human approval required for refunds above INR 1000"}

    return _ledger("issue_refund", order_id, requested_amount)
def escalate_to_human(order_id,refund_inr): return _ledger("escalate_to_human",order_id,refund_inr)
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve an order's status, amount, and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant refund policy information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund based on amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]
def run_agent(
    user_message,
    scenario_id="demo",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None,
    resume_state=None
):
    run_id=run_id or str(uuid.uuid4()); STATE.mkdir(exist_ok=True)
    messages = [
    {
        "role": "system",
        "content": "AGENT: Use tools to verify orders and follow refund policy."
    },
    {
        "role": "user",
        "content": user_message
    }
]
    if resume_state is not None:
        messages = resume_state["messages"]
        tokens = resume_state["tokens_used"]
        steps = resume_state["steps"]
        executed = resume_state["tools_executed"]
        llm_calls = resume_state["llm_calls"]
        start_iteration = resume_state["step_number"]
    else:
        tokens = 0
        steps = []
        executed = []
        llm_calls = 0
        start_iteration = 0

    start = time.monotonic()
    def checkpoint(step):
        obj={"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,"step_number":step,"tools_executed":executed,"steps":steps,"llm_calls":llm_calls}
        (STATE/f"{run_id}.json").write_text(json.dumps(obj,default=str),encoding="utf-8")
    for iteration in range(start_iteration,max_iterations):
        if time.monotonic()-start>=time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
        response=mock_llm.chat(messages); llm_calls+=1; tokens+=response.usage.total_tokens
        if tokens>token_limit:
            return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}
        msg=response.choices[0].message
        if not msg.tool_calls:
            content = msg.content or ""

            result = {
                "status": content.split(":", 1)[0],
                "final_answer": content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }

            checkpoint(len(executed))

            checkpoint_path = STATE / f"{run_id}.json"
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            saved["completed"] = True
            saved["final_result"] = result
            checkpoint_path.write_text(
                json.dumps(saved, default=str),
                encoding="utf-8"
            )

            return result
        tc=msg.tool_calls[0]; args=tc.arguments
        messages.append({"role":"assistant","content":None,"tool_calls":[{"id":tc.id,"name":tc.name,"arguments":args}]})
        observation=TOOL_MAP[tc.name](**args); executed.append(tc.name)
        steps.append({"tool":tc.name,"arguments":args,"observation":observation})
        messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":json.dumps(observation)})
        checkpoint(iteration+1)
        if crash_after is not None and len(executed)==crash_after: raise RuntimeError("Simulated crash after checkpoint")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to a human agent.","steps":steps,"llm_calls":llm_calls,"tokens_used":tokens}

def resume_agent(run_id):
    checkpoint_path = STATE / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as f:
        saved_state = json.load(f)
        if saved_state.get("completed", False):
            return saved_state["final_result"]

    user_message = next(
        message["content"]
        for message in saved_state["messages"]
        if message["role"] == "user"
    )

    return run_agent(
        user_message=user_message,
        scenario_id=saved_state["scenario_id"],
        run_id=run_id,
        resume_state=saved_state
    )