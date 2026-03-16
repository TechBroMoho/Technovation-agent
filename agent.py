"""
agent.py — Real LangChain agent using Claude (via ChatAnthropic).

The LLM genuinely reasons about which tools to call and when —
no regex or hardcoded intent parsing. This is the real LangChain pattern.

Architecture:
    User input
        ↓
    ChatAnthropic (Claude) — reasons about what to do
        ↓
    Tool call (get_available_slots or book_slot)
        ↓
    Tool result fed back to Claude
        ↓
    Claude formats final response
        ↓
    AgentExecutor returns to user
"""

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import get_available_slots, book_slot

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
# Prompt
# ---------------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful coaching session scheduling assistant.
Your job is to help users find and book available coaching time slots.

Coach preference context:
  The coach prefers morning sessions, especially the 10 AM and 11 AM slots.
  When showing available times, always pass preferences='default' to the
  get_available_slots tool so preferred slots appear first.

Guidelines:
- Always check availability before booking
- Confirm with the user before finalizing a booking
- Be concise and friendly
- If a user says 'the first one' or 'that one', refer back to the slots you just listed
- Only book slots the user explicitly confirms
- When presenting slots, gently nudge the user toward the coach's preferred
  times (the ones listed first in the tool results). For example you might say
  "The coach's favorite slots are …" or "The coach especially recommends …".
  Never refuse other times — just highlight the preferred ones first."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# ---------------------------------------------------------------------------
# Agent + Executor
# ---------------------------------------------------------------------------

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class CoachingAgent:
    def __init__(self):
        print("Coaching Scheduling Agent ready. Type 'quit' to exit.\n")

    def respond(self, user_input: str) -> str:
        result = agent_executor.invoke({"input": user_input})
        output = result["output"]
        if isinstance(output, list):
            return " ".join(item.get("text", "") for item in output if isinstance(item, dict))
        return output

    def run_interactive(self):
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                print("Agent: Goodbye! Good luck with your sessions.")
                break

            response = self.respond(user_input)
            print(f"\nAgent: {response}\n")
