import streamlit as st

# Set page config FIRST — must be the first Streamlit command
st.set_page_config(page_title="BERT QA", layout="wide")

from transformers import pipeline

# Load QA pipeline with BERT model
@st.cache_resource
def load_qa_pipeline():
    model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    return pipeline("question-answering", model=model_name, tokenizer=model_name)

nlp = load_qa_pipeline()

# Streamlit UI
st.title("🤖 BERT-based Q&A using HuggingFace")

context = st.text_area("📄 Enter Context", height=200, value="""
Trainer: Deepan Raj;
Session 1: Intro about Arenas & LLMs;
Session 2: Hands-on on LLMs
Session 3: LangChain - Intro
""")

question = st.text_input("❓ Ask a Question", value="What's the session 1?")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        QA_input = {"question": question, "context": context}
        result = nlp(QA_input)
        st.success(f"🧠 **Answer**: {result['answer']}")
        st.caption(f"Score: {result['score']:.2f} | Start: {result['start']}, End: {result['end']}")