import os
from langchain_unstructured import UnstructuredLoader
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from weaviate_client import WeaviateDB
from metadata_processor import prepare_metadata_for_weaviate

# Load and process PDF
loader = UnstructuredLoader(
    file_path="ncert_crop.pdf",
    strategy="hi_res",
    partition_via_api=False,  # Use local processing instead of API
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

print(f"Loaded {len(docs)} document segments from PDF")

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"Split into {len(all_splits)} chunks")

from langchain_core.documents import Document
cleaned_splits = [
    Document(page_content=doc.page_content, metadata=prepare_metadata_for_weaviate(doc.metadata))
    for doc in all_splits
]

# Set up OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=120)
client_manager = WeaviateDB()
weaviate_client = client_manager.get_client()
vector_store = WeaviateVectorStore.from_documents(cleaned_splits, embeddings, client=weaviate_client)
print(f"Added {len(cleaned_splits)} documents to vector store")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Create a tool for PDF retrieval
@tool
def retrieve_from_pdf(query: str):
    """Retrieve information from the PDF document based on a query."""
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

# Create ReAct agent
memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve_from_pdf], checkpointer=memory)

# Interactive conversation function
def chat_with_pdf():
    """Interactive chat interface for querying the PDF"""
    config = {"configurable": {"thread_id": "pdf_chat_123"}}
    
    print("\n" + "="*50)
    print("PDF CHAT INTERFACE")
    print("="*50)
    print("Ask questions about the PDF. Type 'quit' to exit.")
    print()
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\nAgent: ", end="", flush=True)
            
            # Stream the agent's response
            for event in agent_executor.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
                config=config,
            ):
                if "messages" in event and event["messages"]:
                    latest_message = event["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        print(latest_message.content, end="", flush=True)
            
            print("\n")
    finally:
            weaviate_client.close()

# Test the agent with some initial queries
def test_agent():
    """Test the agent with some sample queries"""
    config = {"configurable": {"thread_id": "test_123"}}
    
    test_queries = [
        "What is this document about?",
        "What is the first step in growing a crop?",
        "Why is Paheli worried?",
    ]
    
    print("\n" + "="*50)
    print("AGENT TESTING")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        print("-" * 40)
        
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            config=config,
        ):
            if "messages" in event and event["messages"]:
                latest_message = event["messages"][-1]
                if hasattr(latest_message, 'content'):
                    print(latest_message.content)
        print()

# Run the test first, then start interactive chat
if __name__ == "__main__":
    # First run a quick test
    # test_agent()
    
    # Then start interactive chat
    print("\n" + "="*50)
    print("Starting interactive chat...")
    print("="*50)
    chat_with_pdf()
    