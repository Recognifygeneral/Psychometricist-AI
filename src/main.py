"""CLI entry-point — run the psychometric interview in the terminal.

Usage:
    python -m src.main
    # or via pyproject entry-point:  interview
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from src.workflow import build_graph, FACET_ORDER, MAX_TURNS

load_dotenv()


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         AI Psychometricist — Extraversion Assessment        ║
║                                                             ║
║  Have a natural conversation. There are no right or wrong   ║
║  answers — just be yourself.                                ║
║  Type 'quit' at any time to end the session early.          ║
╚══════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state
    initial_state = {
        "current_facet": FACET_ORDER[0],
        "explored_facets": [],
        "probes_used": [],
        "transcript": "",
        "facet_scores": [],
        "overall_score": 0.0,
        "classification": "",
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

        # Check if we've reached the scorer (done)
        if result.get("classification"):
            # Scorer has produced results — print final report
            print("\n" + "═" * 60)
            print("ASSESSMENT COMPLETE")
            print("═" * 60)
            if messages:
                # The last message from scorer is the summary
                print(messages[-1].content)
            print(f"\nOverall Score : {result.get('overall_score', 'N/A')}")
            print(f"Classification: {result.get('classification', 'N/A')}")
            print("═" * 60)
            break

        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended by user.")
            break

        if user_input.lower() == "quit":
            print("\nEnding session early — moving to scoring…")
            # Force scoring by resuming with a short goodbye
            result = graph.invoke(
                Command(resume="I'd prefer to stop here, thank you."),
                config,
            )
            # Set done to force scorer on next router pass
            # We need to update state and re-invoke
            continue

        # Resume the graph with user input
        result = graph.invoke(Command(resume=user_input), config)


if __name__ == "__main__":
    main()
