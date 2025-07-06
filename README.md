# LangChain RAG (Retrieval-Augmented Generation)

A Python project demonstrating Retrieval-Augmented Generation (RAG) using LangChain and LangGraph. This project implements two different approaches to RAG: a custom graph-based approach and a ReAct agent approach.

## Overview

This project creates a question-answering system that:
1. Loads and processes a blog post about AI agents from Lilian Weng's blog
2. Splits the content into chunks and stores them in a vector database
3. Provides two different RAG implementations:
   - **Custom Graph Approach** (`main.py`): A custom LangGraph implementation with explicit retrieval and generation steps
   - **ReAct Agent Approach** (`agent.py`): Uses LangGraph's prebuilt ReAct agent with retrieval tools

## Features

- **Web Scraping**: Automatically loads content from a specified blog post
- **Text Processing**: Splits content into manageable chunks with overlap
- **Vector Search**: Uses OpenAI embeddings for semantic search
- **Two RAG Implementations**: 
  - Custom graph-based approach with explicit control flow
  - ReAct agent approach with tool-based reasoning
- **Conversation Memory**: Maintains conversation context across interactions
- **Streaming Output**: Real-time response generation

## Prerequisites

- Python 3.13.5 or higher
- OpenAI API key
- Internet connection (for loading the blog post)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-rag
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Custom Graph Approach (`main.py`)

This implementation uses a custom LangGraph with explicit nodes for query processing, retrieval, and response generation.

```bash
uv run main.py
```

The script will:
1. Load the blog post about AI agents
2. Process and chunk the content
3. Run several example queries demonstrating the RAG system

### ReAct Agent Approach (`agent.py`)

This implementation uses LangGraph's prebuilt ReAct agent with a retrieval tool.

```bash
uv run agent.py
```

The script will:
1. Load and process the same blog post
2. Create a ReAct agent with retrieval capabilities
3. Run example queries using the agent

## Project Structure

```
langchain-rag/
├── main.py          # Custom graph-based RAG implementation
├── agent.py         # ReAct agent-based RAG implementation
├── pyproject.toml   # Project dependencies and metadata
├── README.md        # This file
└── .venv/          # Virtual environment (created by uv)
```

## Key Components

### Data Processing
- **WebBaseLoader**: Loads content from the specified blog post
- **RecursiveCharacterTextSplitter**: Splits content into 1000-character chunks with 200-character overlap
- **OpenAIEmbeddings**: Creates embeddings for semantic search

### Vector Store
- **InMemoryVectorStore**: Stores document chunks and their embeddings
- Supports similarity search for retrieving relevant content

### RAG Implementations

#### Custom Graph (`main.py`)
- **query_or_respond**: Determines whether to use retrieval or respond directly
- **tools**: Executes retrieval operations
- **generate**: Generates responses using retrieved content
- **Conditional Edges**: Routes between nodes based on tool usage

#### ReAct Agent (`agent.py`)
- **create_react_agent**: Prebuilt agent with reasoning capabilities
- **retrieve tool**: Custom tool for document retrieval
- **MemorySaver**: Maintains conversation state

## Example Queries

The system can answer questions about:
- AI agents and their capabilities
- Task decomposition methods
- Agent architectures and patterns
- Specific concepts from the loaded blog post

## Configuration

- **Model**: Uses GPT-4o-mini for text generation
- **Embeddings**: Uses text-embedding-3-large for vector embeddings
- **Chunk Size**: 1000 characters with 200-character overlap
- **Retrieval**: Returns top 2-3 most relevant documents

## Dependencies

- `langchain`: Core LangChain functionality
- `langchain-openai`: OpenAI integration
- `langchain-community`: Community components (document loaders)
- `langchain-text-splitters`: Text processing utilities
- `langgraph`: Graph-based workflow orchestration
- `beautifulsoup4`: Web scraping for blog content

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on LangChain and LangGraph frameworks
- Uses content from Lilian Weng's blog post about AI agents
- Inspired by modern RAG patterns and best practices
