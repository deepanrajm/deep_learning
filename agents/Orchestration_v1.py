import os
import streamlit as st
import pandas as pd
import tempfile
import subprocess
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.exceptions import ModelProviderError

# ----------------------
# Custom Python Execution Tool
# ----------------------
@tool(show_result=True, stop_after_tool_call=True)
def run_python_code(code: str) -> str:
    """
    Runs Python code in a temporary file and returns stdout/stderr.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Execution error: {e}"

# ----------------------
# LM Studio Chat Wrapper (sanitize roles)
# ----------------------
def sanitize_messages(messages):
    fixed = []
    for m in messages:
        # Support both Message objects and dicts
        role = getattr(m, "role", m.get("role", "user") if isinstance(m, dict) else "user")
        content = getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
        if role not in ["system", "user", "assistant", "tool"]:
            role = "user"
        fixed.append({"role": role, "content": content})
    return fixed

class LMStudioChat(OpenAIChat):
    def response(self, messages, **kwargs):
        clean = sanitize_messages(messages)
        return super().response(clean, **kwargs)

# ----------------------
# Safe run wrapper
# ----------------------
def safe_run(agent, prompt):
    """
    Wraps agent.run to safely extract content and handle dict or Message responses.
    """
    try:
        resp = agent.run(prompt)
        if isinstance(resp, dict):
            return resp.get("content", str(resp))
        elif hasattr(resp, "content"):
            return resp.content
        else:
            return str(resp)
    except Exception as e:
        return f"Agent run error: {e}"

# ----------------------
# Model Setup
# ----------------------
model = LMStudioChat(
    id="llama-3.2-1b-instruct",  # LM Studio Llama 3.2 1B model ID
    api_key="not-needed",
    base_url="http://localhost:1234/v1"
)

# ----------------------
# Define Agents
# ----------------------
manager = Agent(
    name="Manager",
    model=model,
    instructions="You are a project manager. Break user input into tasks for Developer and Tester.",
    markdown=True
)

developer = Agent(
    name="Developer",
    model=model,
    tools=[run_python_code],
    instructions=(
        "You are a Python developer. Write code for the task and run it using run_python_code. "
        "If errors occur, debug and retry until code runs cleanly. Return final working code."
    ),
    markdown=True
)

tester = Agent(
    name="Tester",
    model=model,
    tools=[run_python_code],
    instructions=(
        "You are a tester. Write unittest code for the provided program. "
        "Run tests using run_python_code and provide a concise pass/fail summary."
    ),
    markdown=True
)

reporter = Agent(
    name="Reporter",
    model=model,
    instructions=(
        "Write a final summary report including project description, manager tasks, developer code, "
        "test results, and recommendations."
    ),
    markdown=True
)

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="LM Studio Agentic Workflow", layout="wide")
st.title("🤖 LM Studio Agentic AI Workflow (Llama 3.2 1B)")

project_desc = st.text_area("Project Description", 
                            "Create a Python function to calculate the factorial of a non-negative integer.")

if st.button("Run Workflow"):
    if not project_desc.strip():
        st.warning("Please enter a project description.")
    else:
        try:
            # Manager
            m_text = safe_run(manager, project_desc)
            st.subheader("📋 Manager Output")
            st.markdown(m_text)

            # Developer
            dev_text = safe_run(developer, m_text)
            st.subheader("💻 Developer Output (Code + Execution)")
            st.code(dev_text)

            # Tester
            test_prompt = f"Code:\n{dev_text}"
            test_text = safe_run(tester, test_prompt)
            st.subheader("✅ Tester Output (Tests & Results)")
            st.markdown(test_text)

            # Reporter
            rep_input = f"""
**Project:**  
{project_desc}

**Manager Tasks:**  
{m_text}

**Developer Output:**  
{dev_text}

**Tester Results:**  
{test_text}
"""
            rep_text = safe_run(reporter, rep_input)
            st.subheader("📄 Final Report")
            st.markdown(rep_text)

            # Save summary CSV
            summary = {
                "Project": project_desc,
                "Manager Tasks": m_text,
                "Developer Output": dev_text,
                "Tester Results": test_text,
                "Final Report": rep_text
            }
            df = pd.DataFrame([summary])
            df.to_csv("summary.csv", index=False)
            st.success("Saved summary.csv")
            st.download_button("Download Summary CSV", open("summary.csv").read(), "summary.csv")

        except ModelProviderError as e:
            st.error(f"Model provider error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.code(repr(e))
