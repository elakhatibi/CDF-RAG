import openai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import CONFIG

# Schema for LangGraph
class QueryState(TypedDict):
    query: str

# LLM Query Refiner
class QueryRefiner:
    def __init__(self):
        self.model = CONFIG["QUERY_REFINEMENT_MODEL"]
    
    def refine(self, query: str) -> str:
        prompt = f"Refine the following query for better retrieval:\n{query}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Refine complex queries for enhanced information retrieval."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()

class QueryRefinementAgent:
    def __init__(self):
        self.refiner = QueryRefiner()
    
    def optimize_query(self, state: QueryState) -> QueryState:
        state["query"] = self.refiner.refine(state["query"])
        return state

# ✅ Build the LangGraph using the new builder API
def build_query_refiner_graph():
    builder = StateGraph(QueryState)
    builder.add_node("refine_query", QueryRefinementAgent().optimize_query)
    builder.set_entry_point("refine_query")
    builder.add_edge("refine_query", END)
    return builder.compile()
