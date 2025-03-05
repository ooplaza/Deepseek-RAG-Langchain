# 📄 RAG-Based PDF QA System with Streamlit & LangChain

This project is a **PDF-based Question-Answering System** built using **Streamlit**, **LangChain**, and **Ollama LLM**. It allows users to upload PDF files, index their content as vector embeddings, and retrieve relevant information to answer user queries using **DeepSeek-R1 (1.5B)**.

## 🚀 Features

- 📂 **Upload PDFs** and extract text content.
- 🔍 **Chunk & Index** documents using **vector embeddings**.
- 🧠 **Retrieve relevant document chunks** via similarity search.
- 🤖 **Generate AI-powered answers** using **Ollama LLM**.
- 🖥️ **Interactive chat interface** via Streamlit.

## 📦 Dependencies

Ensure you have the following Python packages installed:

```sh
pip install -r requirements/dev.txt
```

### **📝 Python Version**

This project is compatible with **Python 3.9+**. It is recommended to use a virtual environment to avoid dependency conflicts:

```sh
virutalenv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **🛠 Installing Ollama Locally**

To run the LLM model locally, you need to install **Ollama** on your machine. Follow these steps:

#### **For WSL/Linux users:**

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

#### **For Windows:**

1. Download the installer from [Ollama's official website](https://ollama.com/).
2. Run the installer and follow the setup instructions.

Once installed, verify by running:

```sh
ollama --version
```

### **⚡ Performance Tip**

To prevent your device from overheating 🤣, **select the distilled version of the LLM** with lesser parameters like _[deepseek-r1:1.5b](https://ollama.com/library/deepseek-r1:1.5b)_ for a more efficient experience.

## 🛠️ How It Works

### **1. Upload PDF**

- The uploaded file is saved in `./uploaded/pdfs/`.

### **2. Extract & Process Text**

- The PDF is loaded using **LangChain's PDFPlumberLoader**.
- Text is split into **manageable chunks** using `RecursiveCharacterTextSplitter`.

### **3. Vectorization & Indexing**

- The **DeepSeek-R1 (1.5B) model** converts text chunks into **vector embeddings**.
- These embeddings are stored using **InMemoryVectorStore** (or can be replaced with **ChromaDB** for persistence).

### **4. Query Processing & Document Retrieval**

- When a user asks a question, the system retrieves **relevant text chunks** from indexed documents.

### **5. AI-Powered Answer Generation**

- The retrieved text is **fed into Ollama LLM**, which generates a **concise answer**.

## 📜 Code Overview

### **1️⃣ Upload & Save PDF**

```python
def upload_file(file):
    with open(pdfs_directory + file.name, "wb") as new_file:
        new_file.write(file.getbuffer())
```

### **2️⃣ Load PDF Content**

```python
def load_file(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents
```

### **3️⃣ Chunk & Index Documents**

```python
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)
```

```python
def index_documents(documents):
    vector_store.add_documents(documents)
```

### **4️⃣ Retrieve Relevant Documents**

```python
def retrieve_documents(query):
    return vector_store.similarity_search(query)
```

### **5️⃣ Generate Answers Using Ollama LLM**

```python
def answer_question(question, documents):
    prompt = ChatPromptTemplate.from_template(template)
    context = "\n\n".join([doc.page_content for doc in documents])
    chain = prompt | model  # Chaining prompt to the model
    return chain.invoke({"question": question, "context": context})
```

## 🏃 Running the App

Launch the Streamlit app with:

```sh
streamlit run app.py
```

## 🔮 Future Improvements

- 🔄 **Persist vector embeddings** using ChromaDB instead of `InMemoryVectorStore`.
- 🚀 **Support multiple PDFs** instead of just one per session.
- 🤝 **Integrate additional LLMs** for enhanced responses.
- 📝 **Improve UI/UX** with better chat history and document previews.

## 🤖 Authors & Credits

- Built with **LangChain** & **Streamlit**
- LLM powered by **Ollama (DeepSeek-R1: 1.5B)**
