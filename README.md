# Coaching Scheduling Agent

A LangChain agent that helps users find and book coaching sessions.
Built with real LangChain tool-calling — Claude (via Anthropic API) reasons
about which tools to invoke and when.

## Architecture

```
User input
    ↓
ChatAnthropic (Claude Haiku) — reasons about what to do
    ↓
Tool call: get_available_slots() or book_slot()
    ↓
Tool result fed back to Claude
    ↓
Claude formats final response
    ↓
AgentExecutor returns to user
```

### Files

| File | Purpose |
|------|---------|
| `tools.py` | LangChain `@tool` functions — the actions the agent can take |
| `agent.py` | Agent setup: LLM, prompt, memory, AgentExecutor |
| `main.py` | Interactive CLI entry point |
| `test_agent.py` | Automated demo conversations |

## Setup

1. **Install dependencies**
   ```bash
   pip install langchain langchain-anthropic python-dotenv
   ```

2. **Add your API key**
   ```bash
   cp .env.example .env
   # Edit .env and paste your Anthropic API key
   ```

3. **Run interactively**
   ```bash
   python main.py
   ```

4. **Run demo scenarios**
   ```bash
   python test_agent.py
   ```

## Example Conversation

```
You: Find me available coaching times tomorrow afternoon
Agent: [calls get_available_slots(day="tomorrow", time_of_day="afternoon")]
Agent: Here are the available afternoon slots for tomorrow: Fri 12PM, Fri 2PM.
       Would you like to book one?

You: Book the first one
Agent: [calls book_slot(slot="Fri 12PM")]
Agent: ✅ Booked! You're confirmed for Fri 12PM.
```

## How Tool-Calling Works

The LLM doesn't just pattern-match — it reads the tool docstrings and decides
autonomously which tool to call based on context. This is the key difference
from a rule-based system: the reasoning is emergent, not hardcoded.

`verbose=True` in `AgentExecutor` prints the tool calls in real-time,
which is great for demonstrating this during the team presentation.

## Swapping the LLM

To use a different model, change one line in `agent.py`:

```python
# Groq (free tier)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

Everything else stays the same — that's the power of LangChain's abstraction.
