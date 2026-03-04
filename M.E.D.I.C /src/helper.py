from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os


SOURCE_CONFIG = {
    "Medical_books":  "Medical Book",
    "WHO_Guidelines": "WHO Guideline",
}


def load_pdf_file(data):
    all_documents = []

    subfolders = [
        f for f in os.listdir(data)
        if os.path.isdir(os.path.join(data, f))
    ]

    if subfolders:
        for folder in subfolders:
            folder_path = os.path.join(data, folder)
            source_type = SOURCE_CONFIG.get(folder, folder.replace("_", " ").title())

            # Load each PDF file individually for better metadata control
            pdf_files = [
                f for f in os.listdir(folder_path)
                if f.endswith(".pdf")
            ]

            folder_docs = []
            for pdf_file in pdf_files:
                pdf_path  = os.path.join(folder_path, pdf_file)
                book_name = pdf_file.replace(".pdf", "").strip()

                try:
                    loader = PyPDFLoader(pdf_path)
                    docs   = loader.load()
                except Exception as e:
                    print(f"Warning: Could not load {pdf_path}: {e}")
                    continue

                # Attach metadata to every single page immediately after loading
                for doc in docs:
                    doc.metadata["source_type"] = source_type
                    doc.metadata["book_name"]   = book_name
                    doc.metadata["source"]      = pdf_path

                folder_docs.extend(docs)
                print(f"  Loaded '{book_name}' — {len(docs)} pages [{source_type}]")

            all_documents.extend(folder_docs)
            print(f"Folder '{folder}' done — {len(folder_docs)} pages total\n")

    else:
        # Fallback — flat Data/ folder
        pdf_files = [f for f in os.listdir(data) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path  = os.path.join(data, pdf_file)
            book_name = pdf_file.replace(".pdf", "").strip()
            try:
                loader = PyPDFLoader(pdf_path)
                docs   = loader.load()
            except Exception as e:
                print(f"Warning: Could not load {pdf_path}: {e}")
                continue
            for doc in docs:
                doc.metadata["source_type"] = "Medical Book"
                doc.metadata["book_name"]   = book_name
                doc.metadata["source"]      = pdf_path
            all_documents.extend(docs)
            print(f"  Loaded '{book_name}' — {len(docs)} pages [Medical Book]")

    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Total text chunks: {len(text_chunks)}")
    return text_chunks


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source":      doc.metadata.get("source",      "Unknown"),
                    "page":        doc.metadata.get("page",        0),
                    "source_type": doc.metadata.get("source_type", "Medical Book"),
                    "book_name":   doc.metadata.get("book_name",   "Unknown"),
                }
            )
        )
    return minimal_docs


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings