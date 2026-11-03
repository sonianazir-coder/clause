from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore


DATA_PATH = "shop_handbook.pdf"

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:1b"

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 3


# Indexing

# 1. Load PDF
loader = PyPDFLoader(DATA_PATH)
documents = loader.load()


# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)


# 3. Embed
embedd = init_embeddings(
    model=EMBEDDING_MODEL,
    provider=EMBEDDING_PROVIDER
)


# 4. Store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedd
)


# Retrieval-Augmented Generation (RAG)

# 5. LLM
llm = init_chat_model(
    model=CHAT_MODEL,
    model_provider=CHAT_PROVIDER,
    temperature=0.1
)


# 6. User Query
query = input("You: ").strip()


# 7. Retrieve
results = vector_store.similarity_search_with_score(
    query,
    k=TOP_K
)


# 8. Join retrieved documents
context = "\n\n".join(
    doc.page_content for doc, score in results
)


# 9. Generate
messages = [
    {
        "role": "system",
        "content": "You are a helpful shop assistant. Answer only using the provided context. If the answer is not available, say you don't know."
    },
    {
        "role": "user",
        "content": f"Context: {context}\n\nQuestion: {query}"
    }
]

response = llm.invoke(messages)

print("\nAssistant:", response.content)