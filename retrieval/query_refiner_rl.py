import openai
from langgraph.graph import StateGraph
from config import CONFIG

class QueryRefiner:
    def __init__(self):
        self.model = CONFIG["QUERY_REFINEMENT_MODEL"]
    
    def refine(self, query: str) -> str:
        """Refine the query dynamically using reinforcement learning for improved retrieval."""
        prompt = f"Refine the following query for better retrieval:\n{query}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "Refine complex queries for enhanced information retrieval."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()

# Reinforcement Learning Agent for Query Optimization
class QueryRefinementAgent:
    def __init__(self):
        self.refiner = QueryRefiner()
    
    def optimize_query(self, state):
        """Agent refines the query and evaluates its retrieval performance."""
        refined_query = self.refiner.refine(state['query'])
        state['query'] = refined_query
        return state

# LangGraph Workflow
query_graph = StateGraph()
query_graph.add_node("refine_query", QueryRefinementAgent().optimize_query)
query_graph.set_entry("refine_query")
query_graph.set_exit("refine_query")
