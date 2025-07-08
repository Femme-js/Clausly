class RetrieverNode:
    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}