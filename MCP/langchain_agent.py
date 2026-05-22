import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent   # ✅ correct helper in v1.3.1

MCP_SERVER_URL = "http://127.0.0.1:8888"

def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool on the MCP server via HTTP POST."""
    url = f"{MCP_SERVER_URL}/call/{tool_name}"
    try:
        response = requests.post(url, json=arguments, timeout=10)
        response.raise_for_status()
        return response.json().get("result", "No result found in response.")
    except Exception as e:
        return f"ERROR: {e}"

# Define tools
tools = [
    Tool(
        name="add_note_to_file",
        func=lambda content_str: call_mcp_tool("add_note_to_file", {"content": content_str}),
        description="Add a new note or append text to the notes file."
    ),
    Tool(
        name="read_notes_file",
        func=lambda _: call_mcp_tool("read_notes_file", {}),
        description="Read the current content of the notes file."
    )
]

# Define LLM
llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    temperature=0.7,
    openai_api_base="http://localhost:1234/v1",  # ✅ adjust to your server
    openai_api_key="dummy",
    request_timeout=60,
)

# Build agent using LangGraph’s prebuilt REACT agent
agent = create_react_agent(llm, tools)

if __name__ == "__main__":
    print("🤖 Agent is ready to chat!")
    print("   You can ask it to 'add a note' or 'read my notes'.")
    print("   Type 'exit' or 'quit' to end the session.")

    while True:
        user_query = input("\n> ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # Send user input to agent
            result = agent.invoke({"messages": [("user", user_query)]})

            # Extract the latest AI reply
            messages = result.get("messages", [])
            ai_messages = [m for m in messages if m.__class__.__name__ == "AIMessage"]
            if ai_messages:
                reply = ai_messages[-1].content
            else:
                reply = str(result)

            print(f"\n🤖 Agent: {reply}")

        except Exception as e:
            print(f"An error occurred: {e}")
            break
