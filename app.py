import streamlit as st

from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Local RAG Assistant")
st.write("Ask questions over your local documents using LlamaIndex + Ollama.")


@st.cache_resource
def load_query_engine():
    Settings.llm = Ollama(
        model="llama3.2:3b",
        request_timeout=120.0,
        context_window=2048
    )

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text"
    )

    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    qa_prompt = PromptTemplate(
        """
You are a helpful RAG assistant.

Use the context below to answer the user's question.
Your answer should be based primarily on the retrieved context.

If the context is partially relevant, summarize what can be answered from the context.
If the context is completely irrelevant, say:
"The retrieved documents do not contain enough information to answer this question."

Do not pretend to be the author of the documents.
Do not make up specific facts that are not supported by the context.

Context:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer:
"""
    )

    return index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
        text_qa_template=qa_prompt
    )


query_engine = load_query_engine()

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        response = query_engine.query(question)

    st.subheader("Answer")
    st.write(str(response))

    st.subheader("Sources")
    for i, node in enumerate(response.source_nodes, start=1):
        file_name = node.node.metadata.get("file_name", f"Source {i}")
        with st.expander(f"Source {i}: {file_name}"):
            st.write("Metadata:")
            st.write(node.node.metadata)

            st.write("Text:")
            st.write(node.node.text[:1000])