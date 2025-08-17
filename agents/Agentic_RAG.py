import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document as LlamaDocument
from langchain.docstore.document import Document

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- Setup ---
st.set_page_config(page_title="Self-RAG Iterative Demo", layout="centered")
st.title("🤖 Self-RAG with Iterative Refinement")

# --- Cache & load retriever ---
@st.cache_resource(show_spinner="Indexing PDF...")
def load_retriever(pdf_path: str):
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    llama_docs = [LlamaDocument(text=doc.page_content) for doc in docs]

    hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    node_parser = SemanticSplitterNodeParser(embed_model=hf_embedding)
    nodes = node_parser.get_nodes_from_documents(llama_docs)

    documents = [Document(page_content=node.text, metadata={"source": f"chunk_{i}"}) for i, node in enumerate(nodes)]

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retriever = load_retriever(r"Sample_Document.pdf")

# --- LLM setup ---
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm_studio",
    model="llama-3.2-1b-instruct",
    temperature=0
)

# --- Prompts ---
query_gen_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are an expert query generator.

Based on the conversation history and the latest question below, generate a concise search query to find relevant documents.

Conversation History:
{history}

Latest Question:
{question}

Search Query:
"""
)

answer_gen_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question below. Be concise and precise.

Context:
{context}

Question:
{question}

Answer:
"""
)

query_generator = LLMChain(llm=llm, prompt=query_gen_prompt)
answer_generator = LLMChain(llm=llm, prompt=answer_gen_prompt)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# --- Helpers ---
def format_response_text(text: str) -> str:
    lines = text.strip().split("\n")
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "•", "→")) or line.lower().startswith("step"):
            formatted_lines.append(f"- {line}")
        else:
            formatted_lines.append(f"{line}")
    return "\n\n".join(formatted_lines)

def get_conversation_history():
    # Concatenate last few turns for context (limit to last 6 messages)
    history_text = ""
    history = st.session_state.chat_history[-6:]
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        history_text += f"{role.capitalize()}: {content}\n"
    return history_text

def iterative_self_rag(question, max_iters=2):
    conversation_history = get_conversation_history()
    last_answer = None
    last_query = None
    retrieved_docs = None

    for i in range(max_iters):
        # Generate retrieval query
        query_input = {
            "history": conversation_history,
            "question": question if i == 0 else last_answer
        }
        search_query = query_generator.run(**query_input).strip()

        # Retrieve documents
        docs = retriever.get_relevant_documents(search_query)
        combined_context = "\n\n".join(doc.page_content for doc in docs)

        # Generate answer
        answer = answer_generator.run(context=combined_context, question=question)

        # Update for next iteration
        last_answer = answer
        last_query = search_query
        retrieved_docs = docs

        # Update conversation history for next iteration - include answer as context
        conversation_history += f"Assistant: {answer}\n"

    return last_answer, last_query, retrieved_docs

# --- UI & Chat Loop ---
def chat_interface():
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask something about 3D Measurement...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Running iterative Self-RAG..."):
            answer, search_query, docs = iterative_self_rag(user_input, max_iters=2)

        # Format answer nicely
        formatted_answer = format_response_text(answer)

        # Format citations
        citations_md = "### 📄 Retrieved Document Snippets:\n"
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:250].replace("\n", " ") + "..."
            citations_md += f"- **Source chunk_{i}**: {snippet}\n"

        full_response = f"**Search Query:** `{search_query}`\n\n**Answer:**\n{formatted_answer}\n\n{citations_md}"

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(full_response)

        # Feedback section
        if st.button("👍 Helpful"):
            st.session_state.feedback[user_input] = "helpful"
            st.success("Thanks for your feedback! 😊")

        if st.button("👎 Needs Improvement"):
            st.session_state.feedback[user_input] = "needs_improvement"
            st.info("You can ask a follow-up question or clarify your request.")

chat_interface()
