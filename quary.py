from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=120.0,
    context_window=2048
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

print("Loading index...")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=4
)

question = input("Ask a question: ")

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nSources:")
for i, node in enumerate(response.source_nodes, start=1):
    print(f"\nSource {i}:")
    print(node.node.metadata)
    print(node.node.text[:500])