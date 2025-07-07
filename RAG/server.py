from flask import Flask, request, jsonify
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Step 1: Load documents and build retriever
pdf_path = r"C:\Users\deepa\OneDrive\Documents\GitHub\deep_learning\RAG\Basic_Home_Remedies.pdf"
loader = PDFPlumberLoader(pdf_path)
docs = loader.load()

# Optional: attach dummy source metadata
for doc in docs:
    doc.metadata["source"] = "Home_Remedies"

text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

embedder = HuggingFaceEmbeddings()
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 2: Define prompts and LLM
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm_studio",
    model="llama-3.2-1b-instruct"
)

template = """
1. You are a helpful assistant.
2. Use the following pieces of context to answer the question at the end.
3. Answer only by using the context and articulate it better.
4. Keep the answer crisp and limited to 3–4 sentences.

Context: {context}

Question: {question}

Answer to the question:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="Context:\n{page_content}"
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Step 3: Flask route
@app.route('/rag_query', methods=['POST'])
def rag_query():
    user_question = request.json.get('prompt', '')
    if not user_question:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = qa(user_question)
        return jsonify({"result": result["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
