# 🤖 Conversational RAG PDF Chat App

An **AI-powered Conversational RAG (Retrieval-Augmented Generation) application** that allows users to upload PDFs and ask questions in natural language. The app retrieves relevant information from documents and generates accurate, context-aware answers using LLMs.

---

## 🚀 Features

- Upload and process multiple PDF documents  
- Ask questions in natural language  
- Context-aware responses using chat history  
- Semantic search using vector embeddings  
- Fast LLM inference with Groq  
- Interactive Streamlit UI  

---

## 🧠 Tech Stack

- Python  
- LangChain  
- ChromaDB (Vector Store)  
- HuggingFace Embeddings  
- Groq LLM  
- Streamlit  

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Environment Variables

Create a `.env` file and add:

```bash
HF_TOKEN=your_huggingface_token
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 🔍 How It Works

1. Upload PDF files  
2. Text is extracted and split into chunks  
3. Chunks are converted into embeddings  
4. Stored in a vector database (ChromaDB)  
5. User query retrieves relevant chunks  
6. LLM generates a context-aware response  

---

## 💡 Key Highlights

- Retrieval-Augmented Generation (RAG) pipeline  
- History-aware retriever for better context  
- Session-based chat memory  
- End-to-end LLM application  

---

## 🎯 Use Cases

- Research & Study Assistant  
- Document Analysis  
- Business Knowledge Base  
- Legal / Policy Document Q&A  

---

## 🔮 Future Improvements

- Support more file types (TXT, DOCX)  
- Improve UI/UX  
- Add authentication  
- Cloud deployment (AWS / Streamlit Cloud)  

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Commit your changes  
4. Submit a Pull Request  

---

## 📜 License

This project is open-source. Add your preferred license here.

---

## 🌟 Acknowledgements

- LangChain  
- HuggingFace  
- Streamlit  
