# tool_server.py  ──  FastAPI MCP-style Tool Server
import os
import math
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI(title="ToolServer")


NOTES_FILE_PATH = r"C:\Deepan_Workspace\My_Works\Git_Codes\deep_learning\MCP\notes.txt"

def add_note_to_file(content: str) -> str:
    os.makedirs(os.path.dirname(NOTES_FILE_PATH), exist_ok=True)
    with open(NOTES_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return f"Success: Note appended to {NOTES_FILE_PATH}."

def read_notes_file() -> str:
    try:
        with open(NOTES_FILE_PATH, "r", encoding="utf-8") as f:
            notes = f.read()
        return notes if notes else "The notes file is empty."
    except FileNotFoundError:
        return f"File not found at {NOTES_FILE_PATH}."



def add(a: float, b: float) -> float:
    """Returns a + b"""
    return a + b

def subtract(a: float, b: float) -> float:
    """Returns a - b"""
    return a - b

def multiply(a: float, b: float) -> float:
    """Returns a * b"""
    return a * b

def divide(a: float, b: float) -> float:
    """Returns a / b  (raises on division by zero)"""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

def power(base: float, exponent: float) -> float:
    """Returns base ** exponent"""
    return base ** exponent

def sqrt(a: float) -> float:
    """Returns √a"""
    if a < 0:
        raise ValueError("Cannot take square root of a negative number.")
    return math.sqrt(a)

def modulo(a: float, b: float) -> float:
    """Returns a % b"""
    if b == 0:
        raise ValueError("Modulo by zero is not allowed.")
    return a % b

def absolute(a: float) -> float:
    """Returns |a|"""
    return abs(a)


available_tools = {
    # notes
    "add_note_to_file": add_note_to_file,
    "read_notes_file":  read_notes_file,
    # math
    "add":      add,
    "subtract": subtract,
    "multiply": multiply,
    "divide":   divide,
    "power":    power,
    "sqrt":     sqrt,
    "modulo":   modulo,
    "absolute": absolute,
}


@app.post("/call/{tool_name}")
async def call_tool(tool_name: str, args: Dict[str, Any]):
    if tool_name not in available_tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")
    try:
        result = available_tools[tool_name](**args)
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing '{tool_name}': {e}")



@app.get("/tools")
async def list_tools():
    return {"tools": list(available_tools.keys())}