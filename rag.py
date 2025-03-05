import os

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit Page Config
st.set_page_config(page_title="RAG-based QA", layout="wide")

# Sidebar options
st.sidebar.title("Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Model", ["deepseek-r1:1.5b"]
)  # Add more models here
chunk_size = st.sidebar.slider(
    "Chunk Size", min_value=500, max_value=2000, value=1000, step=100
)

# Initialize Model and Embeddings
model = OllamaLLM(model=selected_model)
embeddings = OllamaEmbeddings(model=selected_model)
vector_store = InMemoryVectorStore(embeddings)

template = """
You are an assistant for question-answering tasks. Use the retrieved context
to answer the question. If you don't know the answer, say so. Keep it concise.
Question: {question}
Context: {context}
Answer:
"""


class RagBasedQA:
    def __init__(self, model, embeddings, vector_store):
        self.model = model
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.pdfs_directory = "./uploaded/pdfs/"
        os.makedirs(self.pdfs_directory, exist_ok=True)

    def upload_file(self, file):
        file_path = os.path.join(self.pdfs_directory, file.name)
        with open(file_path, "wb") as new_file:
            new_file.write(file.getbuffer())
        return file_path

    def load_and_process_file(self, file_path):
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200
        )
        chunked_documents = text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunked_documents)
        return chunked_documents

    def retrieve_documents(self, query):
        return self.vector_store.similarity_search(query)

    def answer_question(self, question, documents):
        prompt = ChatPromptTemplate.from_template(template)
        context = "\n\n".join([doc.page_content for doc in documents])
        chain = prompt | self.model
        return chain.stream({"question": question, "context": context})

    def run(self):
        st.title("📄 RAG-based Question Answering")
        st.write("Upload a PDF and ask questions based on its content.")

        uploaded_file = st.file_uploader(
            "Upload a PDF file", type="pdf", accept_multiple_files=False
        )
        if uploaded_file:
            file_path = self.upload_file(uploaded_file)
            with st.spinner("Hold on, darling… magic takes time! ✨📄"):
                self.load_and_process_file(file_path)
            st.success("Boom! Your document is ready to spill its secrets. 🔥📄")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask a question...")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user", avatar="./logos/froggy.png"):
                st.write(question)

            with st.chat_message("assistant", avatar="./logos/beluga.gif"):
                message_placeholder = st.empty()
                message_placeholder.write("Thinking...")

                related_documents = self.retrieve_documents(question)
                response_stream = self.answer_question(question, related_documents)
                full_response = ""
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.write(full_response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": full_response}
                )


if __name__ == "__main__":
    rag = RagBasedQA(model, embeddings, vector_store)
    rag.run()
