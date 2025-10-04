import requests
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool



MCP_SERVER_URL = "http://127.0.0.1:8888"

def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Calls a tool on the MCP server using a standard HTTP POST request.
    This function is synchronous and robust.
    """
    url = f"{MCP_SERVER_URL}/call/{tool_name}"
    try:
        response = requests.post(url, json=arguments, timeout=10)
        response.raise_for_status() 
        return response.json().get("result", "No result found in response.")
    except requests.exceptions.RequestException as e:
        return f"ERROR: Could not connect to MCP server at {url}. Please ensure mcp_server.py is running. Details: {e}"
    except Exception as e:
        return f"An unknown error occurred: {e}"


tools = [
    Tool(
        name="add_note_to_file",
        func=lambda content_str: call_mcp_tool(
            "add_note_to_file", {"content": content_str}
        ),
        description="Use this to add a new note or append text to the notes file."
    ),
    Tool(
        name="read_notes_file",
        func=lambda _: call_mcp_tool("read_notes_file", {}),
        description="Use this to read the current content of the notes file."
    )
]



llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",  
    temperature=0.7,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",
    request_timeout=60,
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("🤖 Agent is ready to chat!")
    print("   You can ask it to 'add a note' or 'read my notes'.")
    print("   Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:

            user_query = input("\n> ")


            if user_query.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
            

            response = agent.run(user_query)


            print(f"\n🤖 Agent: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")
            break