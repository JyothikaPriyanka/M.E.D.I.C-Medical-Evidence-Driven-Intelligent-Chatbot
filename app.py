from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
app.secret_key = os.urandom(24)
load_dotenv()


PINECONE_API_KEY = os.environ.get('pinecone_api_key')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Lowered threshold from 0.5 to 0.3 to fix empty retrieval
retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)

llm = ChatGroq(model="llama-3.1-8b-instant")

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, "
     "rephrase the question to be standalone and clear. "
     "Do NOT answer it, just rephrase it. "
     "If it's already clear, return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_chat_history():
    history = session.get("chat_history", [])
    messages = []
    for msg in history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def extract_sources(context_docs):
    sources = []
    seen = set()
    for doc in context_docs:
        book_name   = doc.metadata.get("book_name",   None)
        source_type = doc.metadata.get("source_type", None)
        page        = doc.metadata.get("page",        None)

        if not book_name:
            raw_source = doc.metadata.get("source", "Unknown Source")
            book_name  = os.path.basename(raw_source).replace(".pdf", "").strip()

        if not source_type:
            source_type = "Medical Source"

        page_display = int(page) + 1 if page is not None else "N/A"

        key = f"{book_name}-{page_display}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "book":        book_name,
                "page":        page_display,
                "source_type": source_type
            })
    return sources


def is_general_conversation(text):
    general_phrases = [
        "hi", "hello", "hey", "how are you", "good morning",
        "good evening", "good night", "thanks", "thank you",
        "bye", "goodbye", "ok", "okay", "great", "awesome",
        "who are you", "what are you", "what can you do"
    ]
    return text.strip().lower() in general_phrases or \
           any(text.strip().lower() == phrase for phrase in general_phrases)


def save_to_session(msg, answer):
    session.setdefault("chat_history", [])
    session["chat_history"].append({"role": "human", "content": msg})
    session["chat_history"].append({"role": "ai",    "content": answer})
    if len(session["chat_history"]) > 20:
        session["chat_history"] = session["chat_history"][-20:]
    session.modified = True


@app.route("/")
def index():
    session.clear()
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    chat_history = get_chat_history()

    # General conversation — skip RAG, no sources
    if is_general_conversation(msg):
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        answer = response["answer"]
        print("Response (general):", answer)
        save_to_session(msg, answer)
        return jsonify({"answer": answer, "sources": []})

    # Medical question — full RAG with sources
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })

    answer  = response["answer"]
    context = response.get("context", [])

    dont_know_phrases = ["i don't know", "i do not know", "not sure", "no information"]
    answer_used_context = len(context) > 0 and not any(
        phrase in answer.lower() for phrase in dont_know_phrases
    )
    sources = extract_sources(context) if answer_used_context else []

    print("Response:", answer)
    print("Sources:", sources)

    save_to_session(msg, answer)
    return jsonify({"answer": answer, "sources": sources})


@app.route("/clear", methods=["POST"])
def clear():
    session.clear()
    return "cleared", 200


# --- Debug Routes (remove these after confirming badges work) ---

@app.route("/debug", methods=["GET"])
def debug():
    # Check real similarity scores bypassing threshold
    raw_docs = docsearch.similarity_search_with_score("hypertension", k=5)
    results = []
    for doc, score in raw_docs:
        results.append({
            "score":           round(score, 4),
            "source_type":     doc.metadata.get("source_type", "MISSING"),
            "book_name":       doc.metadata.get("book_name",   "MISSING"),
            "page":            doc.metadata.get("page",        "MISSING"),
            "content_preview": doc.page_content[:150]
        })
    return jsonify(results)


@app.route("/debug/stats", methods=["GET"])
def debug_stats():
    # Check how many vectors are actually stored in Pinecone
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("medicalbot")
    stats = index.describe_index_stats()
    return jsonify({"index_stats": str(stats)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)