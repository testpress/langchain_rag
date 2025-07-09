import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from weaviate_client import WeaviateDB
from prompts import system_message_content

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
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Decide whether to answer directly, or call a tool."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # MultiQueryRetriever setup
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
        llm=llm,
    )
    
    @tool
    def retrieve_from_pdf(query: str, k: 3):
        """
        Retrieve information from the PDF document based on a query.
        """
        retrieved_docs = vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized

    retrieve_with_multiquery = Tool(
        name="retrieve_from_pdf",
        func=lambda query: "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in multi_query_retriever.get_relevant_documents(query)
        ),
        description=(
            "Retrieves relevant information from the PDF document based on semantic multi-query search. "
            "Use this for any question that might need evidence or references from the PDF."
            "Decide on how much k to use based on the query."
        ),
    )
    
    memory = MemorySaver()
    
    agent_executor = create_react_agent(
        llm, 
        [retrieve_with_multiquery], 
        prompt=prompt,
        checkpointer=memory,
    )  
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