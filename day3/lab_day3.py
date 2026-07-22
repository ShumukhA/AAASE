"""
============================================================
LAB: FROM PROTOTYPE TO ENTERPRISE
Crossing the Proof-of-Concept Chasm
============================================================

    LAB_STAGE=0 python lab_prototype_to_enterprise.py

  Stage 0  PROTOTYPE        multi-agent graph, happy path only
  Stage 1  ROBUSTNESS       retries, backoff, timeouts, graceful failure
  Stage 2  CONFIG & SECRETS no hardcoded values, .env, Settings object
  Stage 3  OBSERVABILITY    structured JSON logs, latency, run IDs
  Stage 4  GUARDRAILS+COST  input/output validation, token budget
  Stage 5  SERVING          expose the agent as a FastAPI endpoint


============================================================
"""

import json
import logging
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

STAGE = int(os.getenv("LAB_STAGE", "0"))
MOCK = os.getenv("MOCK", "0") == "1"


# ============================================================
# STAGE 2 — CONFIGURATION & SECRETS
# ------------------------------------------------------------
# Prototype behavior (Stage 0-1): values are hardcoded below.
# Enterprise behavior (Stage 2+): everything comes from the
# environment / .env file. Nothing secret lives in the code.
# ============================================================


@dataclass
class Settings:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    request_timeout_s: int = 60
    max_retries: int = 3
    quality_threshold: int = 8       # review score needed to pass
    max_revisions: int = 2           # review -> rewrite loops allowed
    cost_budget_usd: float = 0.25    # Stage 4: hard cap per run
    max_topic_len: int = 120
    log_level: str = "INFO"
    report_style: str = "formal"|"casual"     # Stage 2: formal or casual

    @classmethod
    def from_env(cls) -> "Settings":
        """Enterprise: config is injected, never edited in code."""
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            request_timeout_s=int(os.getenv("REQUEST_TIMEOUT_S", cls.request_timeout_s)),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            quality_threshold=int(os.getenv("QUALITY_THRESHOLD", cls.quality_threshold)),
            max_revisions=int(os.getenv("MAX_REVISIONS", cls.max_revisions)),
            cost_budget_usd=float(os.getenv("COST_BUDGET_USD", cls.cost_budget_usd)),
            max_topic_len=int(os.getenv("MAX_TOPIC_LEN", cls.max_topic_len)),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            report_style=os.getenv("REPORT_STYLE", cls.report_style),
        )


if STAGE >= 2:
    settings = Settings.from_env()
else:
    # Deliberately "prototype-style": tweak by editing source code.
    settings = Settings()



# ============================================================
# STAGE 3 — OBSERVABILITY
# ------------------------------------------------------------
# Prototype: print(). Enterprise: structured JSON logs that a
# platform like Datadog / CloudWatch / Langfuse can index.
# Every run gets a run_id; every LLM call logs node, latency,
# and token usage.
# ============================================================

logger = logging.getLogger("agent")
logger.setLevel(settings.log_level)
_handler = logging.StreamHandler()
if STAGE >= 3:
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
            }
            payload.update(getattr(record, "extra_fields", {}))
            return json.dumps(payload)
    _handler.setFormatter(JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_event(event: str, **fields):
    logger.info(event, extra={"extra_fields": fields})



# ============================================================
# THE MODEL (with a mock for key-free classrooms)
# ============================================================


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 200, "output_tokens": 300}


class FakeChatModel:
    """Offline stand-in so the lab runs without an API key."""

    def __init__(self):
        self.review_calls = 0

    def invoke(self, prompt: str):
        time.sleep(0.2)  # simulate latency
        p = prompt.lower()
        if "score" in p and "reviewer" in p:
            self.review_calls += 1
            # First review fails quality gate -> demonstrates the loop
            score = 6 if self.review_calls == 1 else 9
            return FakeResponse(
                f"SCORE: {score}\nFEEDBACK: Tighten the introduction and add a concrete example."
            )
        if "research" in p:
            return FakeResponse("- Key fact one about the topic\n- Key fact two\n- Key fact three")
        if "summarize" in p:
            return FakeResponse("Concise summary of the research notes.")
        return FakeResponse(
            "INTRODUCTION\nThis report examines the topic in depth, outlining its "
            "background, current relevance, and why it matters to modern organizations.\n\n"
            "BODY\nThe main findings indicate steady growth, meaningful adoption across "
            "industries, and a set of open challenges around governance, integration, "
            "and cost management that practitioners must address deliberately.\n\n"
            "CONCLUSION\nOrganizations that invest early in robust engineering practices "
            "are best positioned to capture the benefits while controlling the risks."
        )


def get_model():
    if MOCK:
        return FakeChatModel()
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=settings.model_name, temperature=settings.temperature)
    if STAGE >= 1:
        kwargs["timeout"] = settings.request_timeout_s  # never hang forever
        kwargs["max_retries"] = 0  # WE own retry logic (see call_llm)
    return ChatOpenAI(**kwargs)


model = get_model()


# ============================================================
# SHARED STATE (the "contract" between agents)
# ============================================================


class ReportState(TypedDict, total=False):
    run_id: str
    topic: str
    research_notes: str
    summary: str
    draft: str
    review_feedback: str
    score: int
    revision_count: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str


# Rough pricing for gpt-4o-mini (USD per 1M tokens) — good
# enough for a budget guardrail; real systems use billing APIs.
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60


class BudgetExceeded(Exception):
    pass



def call_llm(prompt: str, node: str, state: ReportState) -> str:
    if STAGE >= 4:
        if state.get("cost_usd", 0.0) >= settings.cost_budget_usd:
            raise BudgetExceeded(
                f"Cost budget ${settings.cost_budget_usd} exhausted before node '{node}'"
            )

    attempts = settings.max_retries if STAGE >= 1 else 1
    last_err = None
    for attempt in range(1, attempts + 1):
        start = time.time()
        try:
            response = model.invoke(prompt)
            latency = round(time.time() - start, 2)

            usage = getattr(response, "usage_metadata", None) or {}
            t_in = usage.get("input_tokens", len(prompt) // 4)
            t_out = usage.get("output_tokens", len(response.content) // 4)
            state["tokens_in"] = state.get("tokens_in", 0) + t_in
            state["tokens_out"] = state.get("tokens_out", 0) + t_out
            state["cost_usd"] = round(
                state.get("cost_usd", 0.0)
                + t_in * PRICE_IN_PER_M / 1e6
                + t_out * PRICE_OUT_PER_M / 1e6,
                6,
            )

            if STAGE >= 3:
                log_event(
                    "llm_call",
                    run_id=state.get("run_id", "-"),
                    node=node,
                    attempt=attempt,
                    latency_s=latency,
                    tokens_in=t_in,
                    tokens_out=t_out,
                    cost_usd=state["cost_usd"],
                )
            return response.content

        except Exception as exc:  # noqa: BLE001 — chokepoint by design
            last_err = exc
            if attempt == attempts:
                break
            # Exponential backoff with jitter: 1s, 2s, 4s ... +/- noise
            delay = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            if STAGE >= 3:
                log_event(
                    "llm_retry",
                    run_id=state.get("run_id", "-"),
                    node=node,
                    attempt=attempt,
                    error=str(exc)[:200],
                    retry_in_s=round(delay, 2),
                )
            time.sleep(delay)

    raise RuntimeError(f"Node '{node}' failed after {attempts} attempt(s): {last_err}")



# ============================================================
# STAGE 4 — GUARDRAILS (input + output validation)
# ============================================================

INJECTION_PATTERNS = [
    r"ignore (all|previous|the) instructions",
    r"system prompt",
    r"you are now",
    r"pretend to be",
]


def validate_topic(topic: str) -> str:
    """Reject bad input BEFORE spending money on it."""
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic is empty.")
    if len(topic) > settings.max_topic_len:
        raise ValueError(f"Topic too long (max {settings.max_topic_len} chars).")
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValueError("Topic rejected by input guardrail (possible prompt injection).")
    return topic


def validate_report(report: str) -> None:
    """Never ship broken output to a customer."""
    if len(report) < 200:
        raise ValueError("Output guardrail: report suspiciously short.")
    for phrase in ("as an ai language model", "i cannot", "i'm sorry"):
        if phrase in report.lower():
            raise ValueError(f"Output guardrail: refusal artifact found ('{phrase}').")



# ============================================================
# THE AGENTS (LangGraph nodes) — Day 3 material
# ============================================================


def research_node(state: ReportState) -> ReportState:
    notes = call_llm(
        f"You are a research agent. Produce detailed, factual research notes "
        f"as bullet points about: {state['topic']}",
        node="research",
        state=state,
    )
    state["research_notes"] = notes
    return state


def summarize_node(state: ReportState) -> ReportState:
    summary = call_llm(
        f"You are a summarization agent. Summarize these research notes into "
        f"one dense paragraph:\n\n{state['research_notes']}",
        node="summarize",
        state=state,
    )
    state["summary"] = summary
    return state

def write_node(state: ReportState) -> ReportState:
    feedback = state.get("review_feedback", "")
    revision_hint = (
        f"\n\nA reviewer gave this feedback on your previous draft — address it:\n{feedback}"
        if feedback
        else ""
    )
    style_hint = (
        "Write in a casual, conversational tone."
        if settings.report_style == "casual"
        else "Write in a formal, professional tone."
    )
    draft = call_llm(
        f"You are a professional report writer. {style_hint} Write a structured report "
        f"(introduction, body, conclusion) about '{state['topic']}' based on "
        f"this summary:\n\n{state['summary']}{revision_hint}",
        node="write",
        state=state,
    )
    state["draft"] = draft
    return state



def review_node(state: ReportState) -> ReportState:
    verdict = call_llm(
        f"You are a strict quality reviewer. Score this report from 1-10 and "
        f"give one line of feedback. Reply EXACTLY in this format:\n"
        f"SCORE: <number>\nFEEDBACK: <one line>\n\nReport:\n{state['draft']}",
        node="review",
        state=state,
    )
    match = re.search(r"SCORE:\s*(\d+)", verdict)
    state["score"] = int(match.group(1)) if match else 0
    fb = re.search(r"FEEDBACK:\s*(.+)", verdict)
    state["review_feedback"] = fb.group(1).strip() if fb else verdict
    state["revision_count"] = state.get("revision_count", 0) + 1
    if STAGE >= 3:
        log_event(
            "review_verdict",
            run_id=state.get("run_id", "-"),
            score=state["score"],
            revision=state["revision_count"],
        )
    return state


def review_gate(state: ReportState) -> str:
    """Conditional edge: real coordination, not just a pipeline."""
    if state["score"] >= settings.quality_threshold:
        return "approve"
    if state["revision_count"] > settings.max_revisions:
        return "give_up"
    return "revise"


def build_graph():
    g = StateGraph(ReportState)
    g.add_node("research", research_node)
    g.add_node("summarize", summarize_node)
    g.add_node("write", write_node)
    g.add_node("review", review_node)

    g.set_entry_point("research")
    g.add_edge("research", "summarize")
    g.add_edge("summarize", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges(
        "review",
        review_gate,
        {"approve": END, "give_up": END, "revise": "write"},
    )
    return g.compile()


graph = build_graph()




# ============================================================
# RUNNING A REPORT
# ============================================================


def generate_report(topic: str) -> ReportState:
    state: ReportState = {
        "topic": topic,
        "run_id": str(uuid.uuid4())[:8],
        "revision_count": 0,
        "cost_usd": 0.0,
    }

    if STAGE >= 4:
        state["topic"] = validate_topic(topic)

    if STAGE >= 3:
        log_event("run_started", run_id=state["run_id"], topic=state["topic"], stage=STAGE)

    try:
        final = graph.invoke(state)
    except BudgetExceeded as exc:
        final = dict(state)
        final["error"] = str(exc)
        log_event("run_aborted_budget", run_id=state["run_id"], error=str(exc))
        return final
    except RuntimeError as exc:
        # Stage 1+: graceful failure — return a useful partial result
        final = dict(state)
        final["error"] = str(exc)
        if STAGE >= 1:
            print(f"[degraded] Run failed but did not crash: {exc}")
            return final
        raise  # Stage 0 prototype: just explode

    if STAGE >= 4 and "draft" in final:
        validate_report(final["draft"])

    if STAGE >= 3:
        log_event(
            "run_finished",
            run_id=final.get("run_id", "-"),
            score=final.get("score"),
            revisions=final.get("revision_count"),
            tokens_in=final.get("tokens_in"),
            tokens_out=final.get("tokens_out"),
            cost_usd=final.get("cost_usd"),
        )
    return final


def save_report(state: ReportState, filename: str = "final_report.txt") -> None:

  
    out_dir = os.getenv("REPORTS_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("AI GENERATED REPORT\n" + "=" * 60 + "\n\n")
        f.write(f"Topic: {state.get('topic')}\n")
        f.write(f"Run ID: {state.get('run_id')}\n")
        f.write(f"Review score: {state.get('score')}\n")
        f.write(f"Cost (USD): {state.get('cost_usd')}\n\n")
        f.write(state.get("draft") or f"NO REPORT PRODUCED — {state.get('error')}")
    print(f"Saved: {filename}")


# ============================================================
# STAGE 5 — SERVING: the agent becomes a product
# ------------------------------------------------------------



def create_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Report Agent API", version="1.0")

    class ReportRequest(BaseModel):
        topic: str

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": STAGE, "model": settings.model_name, "mock": MOCK}

    @app.post("/report")
    def report(req: ReportRequest):
        try:
            result = generate_report(req.topic)
        except ValueError as exc:  # guardrail rejection -> client error
            raise HTTPException(status_code=422, detail=str(exc))
        if result.get("error"):
            raise HTTPException(status_code=503, detail=result["error"])
        return {
            "run_id": result["run_id"],
            "topic": result["topic"],
            "score": result.get("score"),
            "cost_usd": result.get("cost_usd"),
            "report": result.get("draft"),
        }

    return app




if __name__ == "__main__":
    print(f"=== Lab running at STAGE {STAGE} {'(MOCK model)' if MOCK else ''} ===\n")

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        if STAGE < 5:
            sys.exit("Serving is a Stage 5 capability. Run with LAB_STAGE=5.")
        import uvicorn

        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    else:
        topic = os.getenv("TOPIC", "Artificial Intelligence in Healthcare")
        result = generate_report(topic)
        save_report(result)
        print(f"\nFinal score: {result.get('score')} | revisions: {result.get('revision_count')} "
              f"| cost: ${result.get('cost_usd')}")
