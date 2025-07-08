import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from weaviate_client import WeaviateDB

def connect_to_collection(collection_name):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    # Create embeddings and connect to existing collection
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=120)
    client_manager = WeaviateDB()
    weaviate_client = client_manager.get_client()
    
    # Connect to existing collection
    vector_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name=collection_name,
        embedding=embeddings,
        text_key="text"
    )
    
    print(f"Connected to collection: {collection_name}")
    return vector_store, weaviate_client

def create_chat_agent(vector_store):
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    @tool
    def retrieve_from_pdf(query: str):
        """Retrieve information from the PDF document based on a query."""
        retrieved_docs = vector_store.similarity_search(query, k=10)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized

    memory = MemorySaver()
    agent_executor = create_react_agent(llm, [retrieve_from_pdf], checkpointer=memory)  
    return agent_executor

def chat_with_pdf(collection_name):
    vector_store, weaviate_client = connect_to_collection(collection_name)   
    agent_executor = create_chat_agent(vector_store)
    
    config = {"configurable": {"thread_id": f"pdf_chat_{collection_name}"}}
    
    print("\n" + "="*50)
    print(f"PDF CHAT INTERFACE - Collection: {collection_name}")
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
                        print("\n")
                        print(latest_message.content, end="", flush=True)
            
            print("\n")
    finally:
            weaviate_client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve documents from Weaviate")
    parser.add_argument("--collection", help="Collection name to connect to")
    
    args = parser.parse_args()
    
    collection_name = args.collection
    
    if not collection_name:
        collection_name = getpass.getpass("Enter the collection name to connect to: ").strip()
        
        if not collection_name:
            print("No collection name provided. Exiting.")
            exit(1)
    
    chat_with_pdf(collection_name) 