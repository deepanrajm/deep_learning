from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat_model = ChatOpenAI(openai_api_base = "http://localhost:1234/v1", openai_api_key = "lm_studio", model = "llama-3.2-1b-instruct")


messages = [HumanMessage(content="Explain quantum entanglement in simple terms.")]

# Call the model
response = chat_model(messages)

print("Response:", response.content)