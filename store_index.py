from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone                          
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get('pinecone_api_key')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Step 1 — Load all PDFs
print("Loading PDFs...")
extracted_data = load_pdf_file(data='Data/')

# Step 2 — Verify metadata is attached before splitting
print("\nSample metadata from loaded docs:")
for doc in extracted_data[:3]:
    print(f"  source_type : {doc.metadata.get('source_type', 'MISSING')}")
    print(f"  book_name   : {doc.metadata.get('book_name',   'MISSING')}")
    print(f"  source      : {doc.metadata.get('source',      'MISSING')}")
    print(f"  page        : {doc.metadata.get('page',        'MISSING')}")
    print()

# Step 3 — Split into chunks
print("Splitting into chunks...")
text_chunks = text_split(extracted_data)

# Step 4 — Verify metadata survived the split
print("\nSample metadata after splitting:")
for chunk in text_chunks[:3]:
    print(f"  source_type : {chunk.metadata.get('source_type', 'MISSING')}")
    print(f"  book_name   : {chunk.metadata.get('book_name',   'MISSING')}")
    print(f"  page        : {chunk.metadata.get('page',        'MISSING')}")
    print()

# Step 5 — Load embeddings
print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

# Step 6 — Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Step 7 — Delete old index and recreate fresh
existing_indexes = [i.name for i in pc.list_indexes()]
if index_name in existing_indexes:
    print(f"\nDeleting old index '{index_name}'...")
    pc.delete_index(index_name)
    print("Waiting for deletion to complete...")
    time.sleep(10)
    print("Old index deleted.")

print(f"\nCreating new index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print("Waiting for index to be ready...")
time.sleep(10)

# Step 8 — Upload all chunks with metadata
print("\nUploading embeddings to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("\nDone! All documents indexed successfully.")
print(f"Total chunks uploaded: {len(text_chunks)}")