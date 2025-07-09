import os
import json
from pathlib import Path
from langchain_unstructured import UnstructuredLoader
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_message_content = (
    """You are an AI assistant answering questions based on a PDF document, accessible via the 'retrieve_from_pdf' tool.
    For general conversation (e.g., greetings), reply naturally using general knowledge.
    If the input is about the document or refers to it (e.g., 'what is this about?', 'summarize this', 'what does it say about X?'), use 'retrieve_from_pdf' to respond.
    For broad queries (e.g., 'summarize this document'), call 'retrieve_from_pdf' with a general query like 'main topics and summary'."""
)

# Set up OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=120)
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Configuration file to track indexed PDFs
CONFIG_FILE = "pdf_index_config.json"
CHROMA_DIR = "./chroma_db"

def load_config():
    """Load the configuration file that tracks indexed PDFs"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    """Save the configuration file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_indexed_pdfs():
    """Get list of indexed PDFs"""
    config = load_config()
    return list(config.keys())

def index_pdf(pdf_path):
    """Index a PDF file and store it in Chroma"""
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found!")
        return False
    
    print(f"Loading and processing {pdf_path}...")
    
    # Load and process PDF
    loader = UnstructuredLoader(
        file_path=pdf_path,
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

    # Filter complex metadata that Chroma can't handle
    filtered_splits = []
    for doc in all_splits:
        # Manually filter complex metadata
        filtered_metadata = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered_metadata[key] = value
        doc.metadata = filtered_metadata
        filtered_splits.append(doc)

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=filtered_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=Path(pdf_path).stem  # Use filename without extension as collection name
    )
    print(f"Added {len(all_splits)} documents to Chroma vector store")

    # Update config
    config = load_config()
    config[pdf_path] = {
        "collection_name": Path(pdf_path).stem,
        "chunks": len(all_splits),
        "indexed_at": str(Path(pdf_path).stat().st_mtime)
    }
    save_config(config)
    
    print(f"Successfully indexed {pdf_path}")
    return True

def create_chat_agent(pdf_path):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    """Create a chat agent for a specific PDF"""
    config = load_config()
    if pdf_path not in config:
        print(f"Error: {pdf_path} is not indexed!")
        return None
    
    collection_name = config[pdf_path]["collection_name"]
    
    # Load the existing Chroma collection
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name
    )
    
    # Create a tool for PDF retrieval
    @tool
    def retrieve_from_pdf(query: str, k: int = 10):
        """
        Retrieve information from the PDF document based on a query.
        
        Note: It is important to choose the 'k' parameter based on the query type:
        - For summary/overview questions: use k=15-25 (more documents needed)
        - For broad overview questions: use k=8-12
        - For specific details: use k=3-5  
        - For comprehensive analysis: use k=20-30
        - For document summary: use k=20-30
        - Maximum allowed is k=30
        
        Args:
            query (str): The search query.
            k (int): Number of relevant documents to retrieve (1-30). ALWAYS choose based on query scope.
        """
        print(f"Retrieving {k} documents for query: {query}")
        retrieved_docs = vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized
    
    # Create ReAct agent
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, [retrieve_from_pdf], 
        prompt=prompt, checkpointer=memory)
    
    return agent_executor

def chat_with_pdf(pdf_path):
    """Interactive chat interface for querying a specific PDF"""
    agent_executor = create_chat_agent(pdf_path)
    if not agent_executor:
        return
    
    config = {"configurable": {"thread_id": f"pdf_chat_{Path(pdf_path).stem}"}}
    
    print(f"\n" + "="*50)
    print(f"CHAT INTERFACE - {Path(pdf_path).name}")
    print("="*50)
    print("Ask questions about the PDF. Type 'quit' to exit.")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Collect all events, print only the final response
        final_message = None
        for event in agent_executor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            if "messages" in event and event["messages"]:
                latest_message = event["messages"][-1]
                if hasattr(latest_message, 'content'):
                    final_message = latest_message.content
        if final_message:
            print(f"\nAgent: {final_message}\n")
        else:
            print("\nAgent: (No response)\n")

def show_indexed_pdfs():
    """Display all indexed PDFs"""
    indexed_pdfs = get_indexed_pdfs()
    if not indexed_pdfs:
        print("No PDFs have been indexed yet.")
        return []
    
    print("\nIndexed PDFs:")
    print("-" * 40)
    for i, pdf_path in enumerate(indexed_pdfs, 1):
        config = load_config()
        pdf_info = config[pdf_path]
        print(f"{i}. {Path(pdf_path).name}")
        print(f"   Chunks: {pdf_info['chunks']}")
        print(f"   Indexed: {pdf_info['indexed_at']}")
        print()
    
    return indexed_pdfs

def main_menu():
    """Main interactive menu"""
    while True:
        print("\n" + "="*50)
        print("PDF RAG SYSTEM")
        print("="*50)
        print("1. Index a new PDF")
        print("2. Chat with indexed PDF")
        print("3. View indexed PDFs")
        print("4. Exit")
        print("-" * 50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            pdf_path = input("Enter the path to the PDF file: ").strip()
            if pdf_path:
                index_pdf(pdf_path)
        
        elif choice == "2":
            indexed_pdfs = get_indexed_pdfs()
            if not indexed_pdfs:
                print("No PDFs have been indexed yet. Please index a PDF first.")
                continue
            
            print("\nAvailable PDFs:")
            for i, pdf_path in enumerate(indexed_pdfs, 1):
                print(f"{i}. {Path(pdf_path).name}")
            
            try:
                pdf_choice = int(input(f"\nSelect PDF (1-{len(indexed_pdfs)}): ")) - 1
                if 0 <= pdf_choice < len(indexed_pdfs):
                    selected_pdf = indexed_pdfs[pdf_choice]
                    chat_with_pdf(selected_pdf)
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number!")
        
        elif choice == "3":
            show_indexed_pdfs()
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")

if __name__ == "__main__":
    main_menu()