"""
agent.py — LangGraph agent with coach/participant role detection.

Flow:
  1. First message → agent asks: coach or participant?
  2. Coach  → sends Cal.com OAuth link → asks preferences
  3. Participant → sends Google OAuth link → finds slots → books
"""

import os
from typing import Optional, Dict, List
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from coaching_agent.tools import get_available_slots, book_slot

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
)

tools = [get_available_slots, book_slot]

# ── Base URL — used to build OAuth links the agent sends to users ──────
BASE_URL = (
    os.getenv("BASE_URL")
    or (f"https://{os.getenv('RAILWAY_PUBLIC_DOMAIN')}"
        if os.getenv("RAILWAY_PUBLIC_DOMAIN") else None)
    or "http://localhost:8000"
)

CALCOM_LOGIN_URL = f"{BASE_URL}/auth/calcom/login"
GOOGLE_LOGIN_URL = f"{BASE_URL}/auth/google/login?user_type=participant"

# ── System prompts ─────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are the first point of contact for the Technovation coaching platform.

Your ONLY job right now is to warmly greet the user and ask whether they are a coach or a participant.

Say something like:
"Welcome to the Technovation coaching platform! 👋 Are you a coach or a participant?"

Do not try to help with anything else until they answer."""

COACH_SYSTEM_PROMPT = f"""You are a helpful onboarding assistant for coaches on the Technovation scheduling platform.

Onboard coaches step by step — do NOT skip steps:

STEP 1 — Connect Cal.com (ALWAYS first):
  Tell the coach they need to connect their Cal.com account so participants can book with them.
  Give them this exact link: {CALCOM_LOGIN_URL}
  Wait for them to confirm they've connected before moving on.

STEP 2 — Preferences (only AFTER they confirm Cal.com is connected):
  Ask: "Are you comfortable with participants rebooking or rescheduling sessions? (yes/no)"
  Save their answer.

STEP 3 — Done:
  Tell them they're all set! Their availability from Cal.com will be visible to participants.

Guidelines:
- Be warm and professional
- Never skip steps or combine them
- If they haven't connected Cal.com yet, remind them to do that first before asking preferences
- Keep responses short and clear"""

PARTICIPANT_SYSTEM_PROMPT = """You are a friendly scheduling assistant helping participants book coaching sessions.

Help participants step by step — do NOT skip steps:

STEP 1 — Understand what they need:
  Ask: "What kind of coaching are you looking for? (e.g. Python, Machine Learning, Web Dev, iOS)"

STEP 2 — Find available slots:
  Use get_available_slots to find times. Always show times in the participant's local timezone.
  Present options clearly — e.g. "Monday 10:00 AM PT, Tuesday 2:00 PM PT"

STEP 3 — Confirm and book:
  Once they pick a slot, confirm: "Just to confirm — you'd like to book [time]?"
  Then use book_slot with the exact ISO timestamp.

STEP 4 — Follow-up:
  After booking, tell them:
  - They'll receive a calendar invite
  - You'll follow up after the session to see how it went
  - They can come back to rebook anytime

Guidelines:
- Be warm and encouraging
- Always confirm before booking
- Never book without explicit confirmation
- If a tool fails, explain clearly and suggest trying again"""


def _make_agent(system_prompt: str):
    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


router_agent      = _make_agent(ROUTER_SYSTEM_PROMPT)
coach_agent       = _make_agent(COACH_SYSTEM_PROMPT)
participant_agent = _make_agent(PARTICIPANT_SYSTEM_PROMPT)


# ── Session state ──────────────────────────────────────────────────────

class SessionState:
    def __init__(self):
        self.role: Optional[str] = None       # "coach" | "participant" | None
        self.chat_history: list = []
        self.calcom_connected: bool = False
        self.google_connected: bool = False
        self.preferences_set: bool = False


# ── CoachingAgent ──────────────────────────────────────────────────────

class CoachingAgent:
    def __init__(self):
        self.state = SessionState()

    def _detect_role(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["coach", "mentor", "teacher", "instructor"]):
            return "coach"
        if any(w in t for w in ["participant", "student", "learner",
                                  "book", "schedule", "session", "i want"]):
            return "participant"
        return None

    def _get_agent(self):
        if self.state.role == "coach":
            return coach_agent
        elif self.state.role == "participant":
            return participant_agent
        return router_agent

    def _invoke(self, agent, user_input: str) -> str:
        self.state.chat_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": self.state.chat_history})
        ai_messages = result["messages"]

        response_text = ""
        for msg in reversed(ai_messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                response_text = msg.content
                break
            elif hasattr(msg, "content") and isinstance(msg.content, list):
                texts = [
                    block.get("text", "")
                    for block in msg.content
                    if isinstance(block, dict) and "text" in block
                ]
                if texts:
                    response_text = " ".join(texts)
                    break

        self.state.chat_history = ai_messages
        return response_text

    def respond(self, user_input: str) -> str:
        if self.state.role is None:
            detected = self._detect_role(user_input)
            if detected:
                self.state.role = detected
            return self._invoke(self._get_agent(), user_input)
        return self._invoke(self._get_agent(), user_input)

    def set_calcom_connected(self):
        self.state.calcom_connected = True

    def set_google_connected(self):
        self.state.google_connected = True

    def run_interactive(self):
        print("Technovation Scheduling Assistant. Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("Agent: Goodbye!")
                break
            print(f"\nAgent: {self.respond(user_input)}\n")


# ── SessionManager ─────────────────────────────────────────────────────

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, CoachingAgent] = {}

    def get_or_create(self, session_id: str) -> CoachingAgent:
        if session_id not in self._sessions:
            self._sessions[session_id] = CoachingAgent()
        return self._sessions[session_id]

    def respond(self, session_id: str, user_input: str) -> str:
        return self.get_or_create(session_id).respond(user_input)

    def set_calcom_connected(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].set_calcom_connected()

    def set_google_connected(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].set_google_connected()

    def get_role(self, session_id: str) -> Optional[str]:
        if session_id in self._sessions:
            return self._sessions[session_id].state.role
        return None

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())
