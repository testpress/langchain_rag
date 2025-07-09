#!/usr/bin/env python3
"""
Test script to verify the retrieval chain functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import unstructured_pdf
sys.path.insert(0, str(Path(__file__).parent))

from unstructured_pdf import create_chat_agent, index_pdf

def test_retrieval_chain():
    """Test the retrieval chain functionality"""
    
    # Test PDF path - you can change this to any PDF you have
    test_pdf = "ncert_crop.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"Test PDF {test_pdf} not found. Please ensure you have a PDF file to test with.")
        return
    
    print("Testing retrieval chain functionality...")
    
    # First, index the PDF if it's not already indexed
    print(f"Indexing {test_pdf}...")
    success = index_pdf(test_pdf)
    
    if not success:
        print("Failed to index PDF")
        return
    
    # Create the retrieval chain
    print("Creating retrieval chain...")
    retrieval_chain = create_chat_agent(test_pdf)
    
    if not retrieval_chain:
        print("Failed to create retrieval chain")
        return
    
    # Test the chain with a simple query
    test_query = "What is this document about?"
    print(f"\nTesting with query: '{test_query}'")
    
    try:
        response = retrieval_chain.invoke(test_query)
        print(f"Response: {response}")
        print("\n✅ Retrieval chain test successful!")
    except Exception as e:
        print(f"❌ Error testing retrieval chain: {e}")
        return
    
    # Test with another query
    test_query2 = "Can you summarize the main points?"
    print(f"\nTesting with query: '{test_query2}'")
    
    try:
        response2 = retrieval_chain.invoke(test_query2)
        print(f"Response: {response2}")
        print("\n✅ Second test successful!")
    except Exception as e:
        print(f"❌ Error in second test: {e}")

if __name__ == "__main__":
    test_retrieval_chain() 