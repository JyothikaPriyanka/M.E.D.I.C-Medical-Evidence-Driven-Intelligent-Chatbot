# M.E.D.I.C-Medical-Evidence-Driven-Intelligent-Chatbot
> *"Intelligent Answers. Evidence Based. Always."*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-latest-green?style=for-the-badge)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-MiniLM-orange?style=for-the-badge&logo=huggingface)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-red?style=for-the-badge)

---

## 📌 Problem Statement

Medical knowledge is scattered across thousands of books, WHO guidelines, and clinical documents. Existing AI tools like ChatGPT answer medical questions confidently — but from **unverified, untraceable sources**. In medicine, a single wrong answer about dosage or symptoms can have life-threatening consequences.

**There is no system that combines the speed of AI with the trustworthiness of verified medical literature.**

---

## ✅ Solution

**M.E.D.I.C** is a RAG-powered Medical Chatbot that:
- Answers questions strictly from **verified medical documents**
- Traces every answer back to the **exact book and page number**
- Prevents hallucination through **multi-layer safety checks**
- Supports **natural multi-turn conversations** with history awareness

---

## 🏗️ Architecture

```
User Question
      │
      ▼
History-Aware Retriever
(Rephrases question using chat history)
      │
      ▼
Pinecone Vector Search
(Finds top 3 most relevant medical chunks)
      │
      ▼
Groq LLaMA 3.1
(Generates answer from retrieved chunks only)
      │
      ▼
Hallucination Detection Layer
(Suppresses sources if model is uncertain)
      │
      ▼
Answer + Source Badges
(Book Name + Page Number)
```

---

## 🚀 Features

- 📚 **Document-Grounded Answers** — Every response is based strictly on your uploaded medical PDFs
- 🔍 **Semantic Vector Search** — Finds meaning, not just keywords
- 📖 **Source Traceability** — Shows exact book name and page number for every answer
- 🧠 **Chat History Awareness** — Understands follow-up questions in context
- 🛡️ **Hallucination Prevention** — Says "I don't know" when context is insufficient
- ⚡ **Ultra-Fast Inference** — Powered by Groq's high-speed LLaMA 3.1 engine
- 💬 **General Conversation Handling** — Handles greetings separately without wasting API calls

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Framework** | Flask |
| **LLM** | Groq — LLaMA 3.1 8b Instant |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Database** | Pinecone Serverless (AWS us-east-1) |
| **RAG Framework** | LangChain |
| **PDF Loader** | PyPDFLoader |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
MEDIC/
│
├── src/
│   ├── helper.py          # PDF ingestion, chunking, embedding functions
│   └── prompt.py          # System prompt for the medical assistant
│
├── Data/
│   ├── Medical_books/     # Medical PDF books
│   └── WHO_Guidelines/    # WHO guideline PDFs
│
├── templates/
│   └── index.html         # Chat UI
│
├── store_index.py         # One-time indexing script — builds Pinecone index
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed to Git)
└── README.md
```

---

## ⚙️ How It Works

**Step 1 — Data Ingestion** (`store_index.py`)
- Loads all PDFs from `Data/` folder with subfolders
- Attaches metadata (book name, page number, source type) to every page
- Splits pages into 500-character chunks with 20-character overlap
- Converts chunks to 384-dimensional vectors using HuggingFace
- Uploads all vectors + metadata to Pinecone

**Step 2 — Query & Answer** (`app.py`)
- User asks a medical question
- LangChain rephrases it using chat history context
- Pinecone retrieves top 3 most relevant chunks (similarity threshold: 0.3)
- Groq LLaMA 3.1 generates a concise answer from those chunks only
- Source badges (book + page) are returned alongside the answer

---

## 🔧 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MEDIC.git
cd MEDIC
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```
pinecone_api_key=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add Your Medical PDFs
```
Data/
├── Medical_books/
│   └── your_medical_book.pdf
└── WHO_Guidelines/
    └── your_guideline.pdf
```

### 6. Build the Pinecone Index (Run Once)
```bash
python store_index.py
```

### 7. Run the Application
```bash
python app.py
```

Visit `http://localhost:8080` in your browser 🚀

---
