"""Refund desk on the correct-by-construction harness (design doc section 17).

States: received -> classified -> decided -> responded -> done
Terminals: done / escalated / exhausted.

- classified: model extracts {order_id, amount_cents, confidence};
  order_id and amount_cents are nullable (union with "null") because a
  request may genuinely lack one — nullability is declared in the
  output schema and enforced by the schema path, not a semantic gate.
  Exit gate requires a usable extraction (else bounded correction,
  then escalation).
- decided: PURE handler wrapping the refund policy. Tools issue_refund and
  flag_for_review are legal ONLY in this state (capability gating).
- responded: model drafts the reply; content gate blocks leaks.

Run:  python3 examples/refund_desk.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))

from adapters.events.stores import InMemoryStore
from adapters.gates.checks import ContentGate, PredicateGate, SchemaGate
from adapters.model.models import FnModel
from app.builder import Harness
from core.events import COMPLETED, ESCALATED, GATE_VERDICT, TOOL_INVOKED
from usecases.ports import ModelResult

# ------------------------------------------------------------------ policy
# Pure function. No prompts, no model. This is FR5.


def decide_refund(ctx: dict) -> str:
    amount = ctx.get("amount_cents") or 0  # None (not extracted) -> 0
    confidence = ctx.get("confidence", 0.0)
    if confidence < 0.5:
        return "REVIEW"
    if amount > 20000:          # over $200 -> human review
        return "REVIEW"
    if amount <= 0:
        return "DENY"
    return "APPROVE"


def decide_handler(ctx: dict) -> dict:
    decision = decide_refund(ctx)
    calls = []
    if decision == "APPROVE":
        calls = [{"name": "issue_refund",
                  "args": {"order_id": ctx["order_id"],
                           "amount_cents": ctx["amount_cents"]}}]
    elif decision == "REVIEW":
        calls = [{"name": "flag_for_review",
                  "args": {"order_id": ctx.get("order_id") or "",
                           "reason": "policy-routing"}}]
    return {"event": "decided",
            "data": {"decision": decision,
                     "amount_str": f"${(ctx.get('amount_cents') or 0) / 100:.2f}"},
            "tool_calls": calls}


# ------------------------------------------------------------------- tools
# Idempotent by construction: keyed writes, safe to retry.


def make_tools():
    ledger: list[dict] = []
    review_queue: list[dict] = []

    def issue_refund(args: dict) -> dict:
        entry = {"order_id": args["order_id"],
                 "amount_cents": args["amount_cents"], "status": "issued"}
        ledger.append(entry)
        return entry

    def flag_for_review(args: dict) -> dict:
        entry = {"order_id": args["order_id"], "reason": args["reason"]}
        review_queue.append(entry)
        return {"queued": True, **entry}

    return issue_refund, flag_for_review, ledger, review_queue


# ------------------------------------------------------------ stub model
# Deterministic stand-in: regex extraction + templated drafting. The real
# model plugs in via OpenAIAdapter without touching anything else.


def classify_fn(prompt: str, schema: dict) -> dict:
    orders = re.findall(r"ORD-\d+", prompt)
    amounts = re.findall(r"\$(\d+(?:\.\d{1,2})?)", prompt)
    if len(orders) == 1 and amounts:
        confidence = 0.9
    elif len(orders) > 1:
        confidence = 0.4   # ambiguous: which order?
    else:
        confidence = 0.0
    cents = int(round(float(amounts[0]) * 100)) if amounts else None
    return {"event": "classified",
            "data": {"order_id": orders[0] if orders else None,
                     "amount_cents": cents,
                     "confidence": confidence},
            "tool_calls": []}


def respond_fn(prompt: str, schema: dict) -> dict:
    decision = (re.search(r"Decision:\s*(\w+)", prompt) or [None, "REVIEW"])[1]
    order = (re.search(r"Order:\s*(\S+)", prompt) or [None, "unknown"])[1]
    amount = (re.search(r"Amount:\s*(\S+)", prompt) or [None, ""])[1]
    bodies = {
        "APPROVE": (f"Regarding order {order}: your refund of {amount} has "
                    f"been approved and issued."),
        "REVIEW": (f"Regarding order {order}: your request is with our "
                   f"review team; we'll update you within 2 business days."),
        "DENY": (f"Regarding order {order}: we can't issue a refund for "
                 f"this request."),
    }
    return {"event": "responded",
            "data": {"message": bodies.get(decision, bodies["REVIEW"])},
            "tool_calls": []}


def model_fn(prompt: str, schema: dict) -> dict:
    if prompt.startswith("CLASSIFY"):
        return classify_fn(prompt, schema)
    return respond_fn(prompt, schema)


# ------------------------------------------------------------------ build
CLASSIFY_PROMPT = """CLASSIFY the customer request below.
Return a JSON envelope: {"event": "classified", "data": {"order_id": str or null, "amount_cents": int or null, "confidence": number}, "tool_calls": []}.
Use null for order_id / amount_cents when the request does not contain one.
Request:
$message
$correction"""

RESPOND_PROMPT = """RESPOND to the customer.
Decision: $decision
Order: $order_id
Amount: $amount_str
Return a JSON envelope: {"event": "responded", "data": {"message": str}, "tool_calls": []}.
Do not reveal internal reasoning.
$correction"""


def build(ledger_tools) -> "BuiltHarness":
    issue_refund, flag_for_review = ledger_tools
    extraction_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": ["string", "null"]},
            "amount_cents": {"type": ["integer", "null"]},
            "confidence": {"type": "number"},
        },
        "required": ["order_id", "amount_cents", "confidence"],
    }
    return (
        Harness.define("refund-desk", version="1.0.0")
        .initial("received")
        .state("received")
        .state("classified",
               prompt_template=CLASSIFY_PROMPT,
               output_schema=extraction_schema,
               exit_gates=("usable_extraction",))
        .state("decided", handler=decide_handler,
               tools=("issue_refund", "flag_for_review"))
        .state("responded",
               prompt_template=RESPOND_PROMPT,
               output_schema={"type": "object",
                              "properties": {"message": {"type": "string"}},
                              "required": ["message"]},
               exit_gates=("no_leak",))
        .state("done", terminal=True)
        .state("escalated", terminal=True)
        .state("exhausted", terminal=True)
        .transition("received", "next", "classified")
        .transition("classified", "classified", "decided")
        .transition("decided", "decided", "responded")
        .transition("responded", "responded", "done")
        .escalation_state("escalated")
        .exhausted_state("exhausted")
        .accounts("cash", "refunds_issued")
        .register_ledger_tool(
            "issue_refund", issue_refund, legal_in={"decided"},
            debit_account="refunds_issued", credit_account="cash",
            amount_arg="amount_cents", reference_arg="order_id",
            args_schema={"type": "object",
                         "properties": {
                             "order_id": {"type": "string"},
                             "amount_cents": {"type": "integer"}},
                         "required": ["order_id", "amount_cents"]})
        .register_tool("flag_for_review", flag_for_review, legal_in={"decided"},
                       args_schema={"type": "object",
                                    "properties": {
                                        "order_id": {"type": "string"},
                                        "reason": {"type": "string"}},
                                    "required": ["order_id", "reason"]})
        .register_gate("usable_extraction", PredicateGate(
            lambda s: ((s["data"].get("order_id") and s["data"]["confidence"] >= 0.3),
                       "extraction unusable: missing order_id or low confidence")))
        .register_gate("no_leak", ContentGate(
            required=("order",),
            forbidden=("internal policy", "confidential", "system prompt")))
        .budget(step_limit=25)
        .build()
    )


class Clock:
    def now_ms(self) -> int:
        return 0


# -------------------------------------------------------------------- main
EPISODES = [
    ("clean approval",
     "Hi, order ORD-1001 arrived broken. Please refund $42.50 to my card."),
    ("large amount -> review",
     "Order ORD-1002 was charged twice, refund $500 please."),
    ("ambiguous multi-order -> review",
     "I have ORD-1003 and ORD-1004; the second arrived damaged, refund $30."),
    ("no order id -> escalate",
     "my package never arrived, I want my money back"),
]


def main() -> None:
    issue_refund, flag_for_review, ledger, review_queue = make_tools()
    built = build((issue_refund, flag_for_review))
    for title, message in EPISODES:
        store = InMemoryStore()
        runner = built.runner(FnModel(model_fn), store, Clock())
        outcome = runner.run({"message": message})
        n_tools = sum(1 for e in outcome.events if e.type == TOOL_INVOKED)
        n_gates = sum(1 for e in outcome.events
                      if e.type == GATE_VERDICT and not e.payload["passed"])
        print(f"[{title}]")
        print(f"  final={outcome.final_state} success={outcome.success} "
              f"decision={outcome.context.get('decision')} "
              f"tools={n_tools} gate_failures={n_gates}")
    print(f"\nledger: {ledger}")
    print(f"review queue: {review_queue}")


if __name__ == "__main__":
    main()
