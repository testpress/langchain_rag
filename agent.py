from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
import getpass
from langgraph.checkpoint.memory import MemorySaver
import os
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query from the post"""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

config = {"configurable": {"thread_id": "def2345"}}

input_message = (
    "What is this post about?"
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()


# input_message = (
#     "Check the complete content of the article and summarize it in 100 words"
# )

# for event in agent_executor.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     event["messages"][-1].pretty_print()

# input_message = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )

# for event in agent_executor.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     event["messages"][-1].pretty_print()