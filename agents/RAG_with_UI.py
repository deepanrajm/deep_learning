import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document as LlamaDocument
from langchain.docstore.document import Document

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

# --- TITLE ---
st.set_page_config(page_title="3D Measurement Handbook for Inspection Manager", layout="centered")
st.title("📘 Inspection Manager Handbook")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LOAD & PROCESS PDF ---
@st.cache_resource(show_spinner="Indexing PDF...")
def load_retriever():
    loader = PDFPlumberLoader(r"Sample_Document.pdf")
    docs = loader.load()
    llama_docs = [LlamaDocument(text=doc.page_content) for doc in docs]

    hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    node_parser = SemanticSplitterNodeParser(embed_model=hf_embedding)
    nodes = node_parser.get_nodes_from_documents(llama_docs)
    documents = [Document(page_content=node.text, metadata={"source": f"chunk_{i}"}) for i, node in enumerate(nodes)]

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retriever = load_retriever()

# --- PROMPT ---
prompt = """
1. You are a helpful assistant.
2. Use the following pieces of context to answer the question at the end.
3. Answer only by using the context and articulate it better.
4. Keep the answer crisp and limited to 3-4 sentences.
5. Answer straight with the result; do not start with "Based on the context" or "Here is the answer in a concise and visually formatted way:"
6. the output should be well formated and not like a normal paragraph

Context: {context}

Question: {question}

Answer to the question:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
    llm=ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm_studio",
        model="llama-3.2-1b-instruct"
    ),
    prompt=QA_CHAIN_PROMPT,
    verbose=False
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=False,
    verbose=False
)
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
# --- CHAT UI ---
def chat_interface():
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask something about 3D Measurement...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            result = qa.run(user_input)
        formatted_result = format_response_text(result)
        st.session_state.chat_history.append({"role": "assistant", "content": formatted_result})
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(formatted_result)

chat_interface()
