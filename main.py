"""
main.py — Run the coaching agent interactively.

Setup:
    1. Copy .env.example to .env
    2. Add your Anthropic API key to .env
    3. Run: python main.py

Or to run automated test scenarios:
    python test_agent.py
"""

from coaching_agent import CoachingAgent

if __name__ == "__main__":
    agent = CoachingAgent()
    agent.run_interactive()
