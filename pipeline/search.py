from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.schema import Document

class WikiSearchNode:
    def __init__(self):
        self.wrapper = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))

    def __call__(self, state):
        print("---WIKIPEDIA---")
        question = state["question"]
        result = self.wrapper.invoke({"query": question})
        return {"documents": Document(page_content=result), "question": question}