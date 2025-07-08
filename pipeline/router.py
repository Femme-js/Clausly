from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from typing import Literal

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(...)

class QueryRouter:
    def __init__(self, config):
        llm = ChatGroq(
            groq_api_key=config["groq_api_key"],
            model_name=config["model_name"],
            temperature=0.0,
            max_tokens=512
        )
        system_msg = (
            "You are an expert at routing a user question to a vectorstore or wikipedia.\n"
            "The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.\n"
            "Use the vectorstore for those topics. Otherwise, use wiki-search."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{question}")
        ])
        self.chain = prompt | llm.with_structured_output(RouteQuery)

    def route(self, state):
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.chain.invoke({"question": question})
        print(f"---ROUTE TO {source.datasource.upper()}---")
        return source.datasource