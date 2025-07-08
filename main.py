from langgraph.graph import END, START, StateGraph
from pprint import pprint
from graph import GraphState
from config import get_config
from utils.vectorstore import VectorStoreIndexer
from pipeline.retriever import RetrieverNode
from pipeline.search import WikiSearchNode
from pipeline.router import QueryRouter

config = get_config()

# Setup
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

indexer = VectorStoreIndexer(config)
indexer.index_urls(urls)
retriever = indexer.as_retriever()

# Instantiate nodes
router = QueryRouter(config)
retriever_node = RetrieverNode(retriever)
wikipedia_node = WikiSearchNode()

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("wiki_search", wikipedia_node)
workflow.add_conditional_edges(
    START,
    router.route,
    {"vectorstore": "retrieve", "wiki_search": "wiki_search"}
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

# Compile and run
app = workflow.compile()
response = app.invoke({"question": "What is agent memory?"})
pprint(response)
