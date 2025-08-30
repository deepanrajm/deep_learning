import streamlit as st
from openai import OpenAI
import os
import re

def extract_sop_steps(raw_text):
    """
    Extracts the numbered SOP steps from the model output,
    ignoring any reasoning or <think>...</think> blocks.
    """
    # Remove all <think>...</think> blocks (including newlines inside)
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    # Sometimes model outputs steps separated by newlines or numbered list
    # Extract lines starting with a number + dot or just text lines after cleaning
    lines = cleaned.splitlines()

    # Filter lines that look like steps (e.g., start with number or nonempty lines)
    steps = [line.strip() for line in lines if line.strip() and
             (re.match(r'^\d+\.', line.strip()) or len(line.strip()) > 5)]

    # If no numbered lines found, just return all lines joined
    if not steps:
        return cleaned

    # Join steps nicely
    return "\n".join(steps)


st.set_page_config(page_title="Software Module Development", layout="wide")
st.title("🤖 Agentic AI - Software Module Development")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
)

# ManagerAgent: Define project scope and specs
def manager_agent(project_description):
    prompt = f"""
You are a software project manager. Based on the project description below, write a clear set of feature requirements and divide them into two tasks: one for a programmer to implement and one for a tester to validate.

Project Description:
{project_description}

Output format:
Task for Programmer:
<Task details>

Task for Tester:
<Task details>
"""
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def programmer_agent(programmer_task):
    prompt = f"""
You are an expert Python programmer. Write the Python code implementing the following task precisely. Include comments explaining key parts.

Task:
{programmer_task}
"""
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def tester_agent(tester_task, code):
    prompt = f"""
You are a software tester. Based on the task and the provided Python code, write test cases using the unittest module to verify the code correctness. Then, write a short test report.

Task:
{tester_task}

Code:
{code}

Output:
1. Test cases code.
2. Test report summary.
"""
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def reporter_agent(project_description, manager_output, code, test_output):
    prompt = f"""
You are a technical writer. Prepare a concise final report summarizing:

- Project description
- Manager's defined tasks
- Programmer's code summary (brief)
- Testing approach and results
- Overall recommendations

Inputs:

Project Description:
{project_description}

Manager Tasks:
{manager_output}

Programmer Code:
{code}

Test Output:
{test_output}

Final report:
"""
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def parse_manager_tasks(manager_output):
    prog_task = ""
    test_task = ""
    lines = manager_output.splitlines()
    prog_mode = False
    test_mode = False
    for line in lines:
        if "Task for Programmer" in line:
            prog_mode = True
            test_mode = False
            continue
        elif "Task for Tester" in line:
            test_mode = True
            prog_mode = False
            continue
        if prog_mode:
            prog_task += line.strip() + "\n"
        elif test_mode:
            test_task += line.strip() + "\n"
    return prog_task.strip(), test_task.strip()

# UI input
st.markdown("### Step 1: Enter a short project description")
project_desc = st.text_area("Project Description", 
                            "Create a Python function to calculate the factorial of a non-negative integer.")

if st.button("Run Orchestration Pipeline"):

    total_steps = 4
    progress = st.progress(0)
    step = 0

    # Step 1: Manager
    status_manager = st.empty()
    status_manager.text("Running Manager Agent...")
    manager_out = manager_agent(project_desc)
    manager_out = extract_sop_steps(manager_out)
    step += 1
    progress.progress(step / total_steps)
    status_manager.text("✅ Manager Agent completed.")
    st.markdown("**Manager Agent Output:**")
    st.code(manager_out)

    prog_task, test_task = parse_manager_tasks(manager_out)

    # Step 2: Programmer
    status_prog = st.empty()
    status_prog.text("Running Programmer Agent...")
    code = programmer_agent(prog_task)
    code = extract_sop_steps(code)
    step += 1
    progress.progress(step / total_steps)
    status_prog.text("✅ Programmer Agent completed.")
    st.markdown("**Programmer Agent Output (Code):**")
    st.code(code, language="python")

    # Step 3: Tester
    status_test = st.empty()
    status_test.text("Running Tester Agent...")
    test_output = tester_agent(test_task, code)
    test_output = extract_sop_steps(test_output)
    step += 1
    progress.progress(step / total_steps)
    status_test.text("✅ Tester Agent completed.")
    st.markdown("**Tester Agent Output (Tests & Report):**")
    st.code(test_output)

    # Step 4: Reporter
    status_report = st.empty()
    status_report.text("Running Reporter Agent...")
    final_report = reporter_agent(project_desc, manager_out, code, test_output)
    final_report = extract_sop_steps(final_report)
    step += 1
    progress.progress(step / total_steps)
    status_report.text("✅ Reporter Agent completed.")
    st.markdown("**Final Report:**")
    st.markdown(final_report)
