import os

import chromadb
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import ChatPromptTemplate
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

template = """
You are an assistant for question-answering tasks. Use the retrieved context
to answer the question. If you don't know the answer, say so. Keep it concise.
Question: {question}
Context: {context}
Answer:
"""


class RagBasedQA:
    def __init__(self):
        self.model = OllamaLLM(model=selected_model)
        self.embeddings = OllamaEmbeddings(model=selected_model)
        self.pdfs_directory = "./uploaded/pdfs/"
        self.chroma_path = "./chromadb"
        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.pdfs_directory, exist_ok=True)

        # Initialize Chroma client and vector store
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="rag_collection",
            embedding_function=self.embeddings,
        )

    def upload_file(self, file):
        """Upload a PDF file and create a directory if it doesn't exist then save the file."""
        file_path = os.path.join(self.pdfs_directory, file.name)
        with open(file_path, "wb") as new_file:
            new_file.write(file.getbuffer())
        return file_path

    def load_and_process_file(self, file_path):
        """
        Load a PDF file from the specified directory using PDFPlumber of langchain.
        Each page of the PDF is representing a document.

        Split the text of the PDF documents into characters using RecursiveCharacterTextSplitter of langchain.
        """
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        # Split the text of the PDF documents into characters using RecursiveCharacterTextSplitter of langchain.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200
        )
        chunked_documents = text_splitter.split_documents(documents)

        # Store vector data in chromadb
        self.vector_store.add_documents(chunked_documents)
        return chunked_documents

    def retrieve_documents(self, query):
        """
        Retrieve the documents using InMemoryVectorStore of langchain we can use
        chroma(https://docs.trychroma.com/docs/overview/introduction).
        By using vector_store to retrieve the documents since all the documents are stored there
        """
        # Retrieve the documents from chromadb top 3 similar documents
        return self.vector_store.similarity_search(query, k=3)

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
    rag = RagBasedQA()
    rag.run()
