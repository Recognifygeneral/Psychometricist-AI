"""CLI entry-point — run the psychometric interview in the terminal.

Usage:
    python -m src.main
    # or via pyproject entry-point:  interview
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from src.workflow import build_graph, MAX_TURNS

load_dotenv()


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         AI Psychometricist — Extraversion Assessment        ║
║                                                             ║
║  This is a scientific feasibility study.                    ║
║  Have a natural conversation — there are no right or wrong  ║
║  answers. Just be yourself.                                 ║
║                                                             ║
║  The interview has {max_turns} questions.                          ║
║  Type 'quit' at any time to end the session early.          ║
╚══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER.format(max_turns=MAX_TURNS))

    graph = build_graph()
    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}

    # Initial state — simplified (no facet routing)
    initial_state = {
        "session_id": session_id,
        "probes_used": [],
        "transcript": "",
        "turn_records": [],
        "turn_features": [],
        "scoring_results": {},
        "overall_score": 0.0,
        "classification": "",
        "confidence": 0.0,
        "facet_scores": [],
        "turn_count": 0,
        "max_turns": MAX_TURNS,
        "done": False,
    }

    # First invocation — triggers router → interviewer → human_turn (interrupt)
    result = graph.invoke(initial_state, config)

    while True:
        # After an interrupt, `result` contains the latest state.
        # The last AI message is the interviewer's question.
        messages = result.get("messages", [])
        if messages:
            last_ai = messages[-1]
            print(f"\n🎙️  Interviewer: {last_ai.content}\n")

        # Check if we've reached the scorer (done with classification)
        if result.get("classification"):
            # Scorer has produced results — print final report
            print("\n" + messages[-1].content if messages else "")
            print(f"\nSession ID: {session_id}")
            break

        # Show progress
        turn = result.get("turn_count", 0)
        print(f"  [{turn}/{MAX_TURNS} turns completed]")

        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended by user.")
            break

        if user_input.lower() == "quit":
            print("\nEnding session early — moving to scoring…")
            result = graph.invoke(
                Command(resume="I'd prefer to stop here, thank you."),
                config,
            )
            continue

        if not user_input:
            print("  (Please type a response)")
            continue

        # Resume the graph with user input
        result = graph.invoke(Command(resume=user_input), config)


if __name__ == "__main__":
    main()
