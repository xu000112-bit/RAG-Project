from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# Local LLM: generates answers
Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=120.0
)

# Local embedding model: converts text into vectors
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

print(f"Loaded {len(documents)} documents.")

print("Building index...")
index = VectorStoreIndex.from_documents(documents)

print("Saving index...")
index.storage_context.persist(persist_dir="storage")

print("Done. Index saved to ./storage")