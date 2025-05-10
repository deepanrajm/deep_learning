import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Set page config
st.set_page_config(page_title="LM Studio Chat", layout="centered")

st.title("💬 Chat with LM Studio")

# Input from user
user_input = st.text_area("Enter your message:", height=150)

# Initialize chat model (pointing to LM Studio local server)
chat_model = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm_studio",  # Dummy key, just needs to be present
    model="llama-3.2-1b-instruct"  # Replace with your actual model name if needed
)

# Button to send message
if st.button("Send"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            # Build message list
            messages = [HumanMessage(content=user_input)]
            
            # Get response from model
            try:
                response = chat_model(messages)
                st.markdown("**Response:**")
                st.success(response.content)
            except Exception as e:
                st.error(f"Error communicating with model: {e}")
    else:
        st.warning("Please enter a message first.")
