"""
test_agent.py — Demo conversations for the real LangChain agent.

Requires a valid ANTHROPIC_API_KEY in your .env file.
Run with: python test_agent.py
"""

from agent import CoachingAgent

def run_scenario(title: str, conversation: list[str]):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {title}")
    print(f"{'='*60}\n")

    agent = CoachingAgent()

    for user_msg in conversation:
        print(f"You: {user_msg}")
        response = agent.respond(user_msg)
        print(f"\nAgent: {response}\n")
        print("-" * 40)


if __name__ == "__main__":

    # Assignment spec example
    run_scenario(
        "Assignment Spec Flow",
        [
            "Find me available coaching times tomorrow afternoon",
            "Book the first one",
        ]
    )

    # Multi-turn with vague references
    run_scenario(
        "Vague Follow-ups",
        [
            "Do you have anything on Thursday?",
            "What about just the morning?",
            "Book the second one",
        ]
    )

    print("\n" + "="*60)
    print("  All scenarios complete.")
    print("="*60)
