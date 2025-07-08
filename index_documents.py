import os
from langchain_unstructured import UnstructuredLoader
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from weaviate_client import WeaviateDB
from metadata_processor import prepare_metadata_for_weaviate

def index_documents(pdf_path="ncert_crop.pdf", collection_name=None):
    """
    Index documents into Weaviate for later retrieval.
    
    Args:
        pdf_path (str): Path to the PDF file to index
        collection_name (str): Optional custom collection name
    
    Returns:
        dict: Information about the indexed collection
    """
    print(f"Starting indexing process for: {pdf_path}")
    
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

    # Prepare metadata for Weaviate storage
    cleaned_splits = [
        Document(page_content=doc.page_content, metadata=prepare_metadata_for_weaviate(doc.metadata))
        for doc in all_splits
    ]
    print(f"Prepared metadata for {len(cleaned_splits)} documents")

    # Set up OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=120)
    client_manager = WeaviateDB()
    weaviate_client = client_manager.get_client()
    
    # Create vector store with optional custom collection name
    vector_store = WeaviateVectorStore.from_documents(
        cleaned_splits, 
        embeddings, 
        client=weaviate_client,
        collection_name=collection_name
    )
    
    # Get collection information
    collection_name = vector_store._collection.name
    print(f"Successfully indexed {len(cleaned_splits)} documents")
    print(f"Collection name: {collection_name}")
    
    # Clean up
    weaviate_client.close()
    
    return {
        "collection_name": collection_name,
        "document_count": len(cleaned_splits),
        "pdf_path": pdf_path,
        "status": "success"
    }

def list_collections():
    """List all available collections in Weaviate"""
    client_manager = WeaviateDB()
    weaviate_client = client_manager.get_client()
    
    try:
        collections = weaviate_client.collections.list_all()
        print("\n--- Available Collections ---")
        for collection in collections:
            print(f"Collection: {collection.name}")
            print(f"  Properties: {len(collection.properties)}")
            print(f"  Objects: {collection.objects_count}")
            print()
    except Exception as e:
        print(f"Error listing collections: {e}")
    finally:
        weaviate_client.close()

def delete_collection(collection_name):
    """Delete a specific collection"""
    client_manager = WeaviateDB()
    weaviate_client = client_manager.get_client()
    
    try:
        weaviate_client.collections.delete(collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
    finally:
        weaviate_client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents into Weaviate")
    parser.add_argument("--pdf", default="ncert_crop.pdf", help="Path to PDF file")
    parser.add_argument("--collection", help="Custom collection name")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--delete", help="Delete a specific collection")
    
    args = parser.parse_args()
    
    if args.list:
        list_collections()
    elif args.delete:
        delete_collection(args.delete)
    else:
        # Index documents
        result = index_documents(args.pdf, args.collection)
        print(f"\nIndexing completed!")
        print(f"Collection: {result['collection_name']}")
        print(f"Documents: {result['document_count']}")
        print(f"PDF: {result['pdf_path']}") 