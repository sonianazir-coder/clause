from pathlib import Path

from langchain.chat_models import init_chat_model # LLM
from langchain.embeddings import init_embeddings  # Embeddings

from langchain_core.documents import Document # .txt, .md
# from langchain_community.document_loaders import PyPDFLoader # .pdf

from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore # vector store

DATA_PATH = "shop_handbook.md"
CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:1b"
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text:latest"
TOP_K = 3

# Indexing
# 1. Load
text = Path("shop_handbook.md").read_text(encoding="utf-8") # string object
doc = Document(
    page_content=text, 
    metadata={"source": "shop_handbook.md"})

# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents([doc])

# 3. Embed
embedd = init_embeddings(
    model="nomic-embed-text:latest",
    provider="ollama"
)

# 4. Store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedd
)

# Retrieval-Augmented Generation (RAG)
# LLM
llm = init_chat_model(
    model="gemma3:1b",
    model_provider="ollama",
    temperature=0.1  # factual answers, less guessing
)

query = input("You: ").strip()

# Retrieve
results = vector_store.similarity_search_with_score(query, k=TOP_K) # [(Documents ,float)]
context = "\n\n".join(d.page_content for d in retrieved_docs) # string object

texts = []
for doc in retriever:
    texts.append(doc.page_content)
context = "\n\n".join(texts) # string object                

# Generate
messages = [
    {"role": "system", "content": "You are a helpful Clause shop assistant"},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]
print(llm.invoke(messages).text)