

import os
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

# 1. Create the FastAPI application instance
app = FastAPI(title="ToolServer")

# 3. --- Define Your Tools ---
NOTES_FILE_PATH = r"D:\Petramount\GL\LLMs\test-c3\notes.txt"
print (NOTES_FILE_PATH)

def add_note_to_file(content: str) -> str:
    """Appends the given content to the user's local notes file."""
    try:
        os.makedirs(os.path.dirname(NOTES_FILE_PATH), exist_ok=True)
        with open(NOTES_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return f"Success: Note appended to {NOTES_FILE_PATH}."
    except Exception as e:
        return f"An error occurred: {e}"

def read_notes_file() -> str:
    """Reads and returns the entire content of the user's local notes file."""
    try:
        with open(NOTES_FILE_PATH, "r", encoding="utf-8") as f:
            notes = f.read()
        return notes if notes else "The notes file is empty."
    except FileNotFoundError:
        return f"The notes file does not exist at {NOTES_FILE_PATH}."
    except Exception as e:
        return f"Error reading file: {e}"

# 3. Create a dictionary to hold your functions
# This allows us to call them by name.
available_tools = {
    "add_note_to_file": add_note_to_file,
    "read_notes_file": read_notes_file,
}

# 4. Create the API endpoint that the agent will call
# This endpoint works exactly like the original MCP server's endpoint.
@app.post("/call/{tool_name}")
async def call_tool(tool_name: str, args: Dict[str, Any]):
    """
    Receives a request from the agent, finds the tool by its name,
    and runs it with the provided arguments.
    """
    if tool_name not in available_tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")
    
    tool_function = available_tools[tool_name]
    
    try:
        # Call the function (e.g., add_note_to_file(**{"content": "..."}))
        result = tool_function(**args)
        # Return the result in the format the agent expects
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing tool '{tool_name}': {e}")