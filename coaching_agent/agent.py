"""
agent.py — LangGraph agent using Claude (via ChatAnthropic).

The LLM genuinely reasons about which tools to call and when —
no regex or hardcoded intent parsing.

Architecture:
    User input
        |
    ChatAnthropic (Claude) — reasons about what to do
        |
    Tool call (get_available_slots or book_slot)
        |
    Tool result fed back to Claude
        |
    Claude formats final response
        |
    Agent returns to user
"""

import os
import uuid
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from coaching_agent.tools import get_available_slots, book_slot

# Load API key from .env
load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # fast + cheap, great for tool-calling demos
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,  # deterministic responses for demos
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

tools = [get_available_slots, book_slot]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful coaching session scheduling assistant.
Your job is to help users find and book available coaching time slots.

How it works:
  - The get_available_slots tool returns real availability from Cal.com.
  - Slots are returned as ISO 8601 timestamps (e.g. 2026-04-10T10:00:00Z).
  - Coach-preferred times are listed first in the results.
  - When booking, pass the exact ISO timestamp from the slot listing to book_slot.

Guidelines:
- Always check availability before booking
- Confirm with the user before finalizing a booking
- Be concise and friendly
- If a user says 'the first one' or 'that one', refer back to the slots you just listed
- Only book slots the user explicitly confirms
- When presenting slots, convert ISO timestamps to a human-friendly format
  (e.g. "10:00 AM PT") but keep the original ISO timestamp for booking
- When slots are labeled as "coach-preferred", gently nudge the user toward those
- If a tool returns an error, explain it clearly and suggest the user try again"""

# ---------------------------------------------------------------------------
# Agent (LangGraph)
# ---------------------------------------------------------------------------

react_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class CoachingAgent:
    def __init__(self):
        self.chat_history = []

    def respond(self, user_input: str) -> str:
        self.chat_history.append(HumanMessage(content=user_input))

        result = react_agent.invoke({"messages": self.chat_history})

        # The last message in the result is the AI response
        ai_messages = result["messages"]
        # Find the last AI text message
        response_text = ""
        for msg in reversed(ai_messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                response_text = msg.content
                break
            elif hasattr(msg, "content") and isinstance(msg.content, list):
                texts = [block.get("text", "") for block in msg.content if isinstance(block, dict) and "text" in block]
                if texts:
                    response_text = " ".join(texts)
                    break

        # Update history with full conversation
        self.chat_history = ai_messages

        return response_text

    def run_interactive(self):
        print("Coaching Scheduling Agent ready. Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("Agent: Goodbye! Good luck with your sessions.")
                break

            response = self.respond(user_input)
            print(f"\nAgent: {response}\n")


# ---------------------------------------------------------------------------
# Multi-session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages multiple CoachingAgent instances keyed by session ID."""

    def __init__(self):
        self._sessions: dict[str, CoachingAgent] = {}

    def get_or_create(self, session_id: str) -> CoachingAgent:
        if session_id not in self._sessions:
            self._sessions[session_id] = CoachingAgent()
        return self._sessions[session_id]

    def respond(self, session_id: str, user_input: str) -> str:
        agent = self.get_or_create(session_id)
        return agent.respond(user_input)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())
