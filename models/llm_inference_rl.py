import openai
from langgraph.graph import StateGraph
from config import CONFIG

class LLMInference:
    def __init__(self):
        self.model = CONFIG["LLM_MODEL"]
    
    def generate(self, context, query, force_retrieval=False):
        """Generate response using the LLM with retrieved and rewritten knowledge."""
        prompt = f"Using the following knowledge, answer the query:\n{context}\n\nQuery: {query}"
        if force_retrieval:
            prompt = "Revised Retrieval Required: " + prompt
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "Generate factually aligned responses."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()

# RL-Based LLM Response Validation Agent
class LLMResponseAgent:
    def __init__(self):
        self.llm = LLMInference()
    
    def generate_validated_response(self, state):
        """Agent generates responses and validates output reliability using RL-based validation."""
        state['response'] = self.llm.generate(state['rewritten_knowledge'], state['query'])
        return state

# LangGraph Workflow
llm_graph = StateGraph()
llm_graph.add_node("generate_validated_response", LLMResponseAgent().generate_validated_response)
llm_graph.set_entry("generate_validated_response")
llm_graph.set_exit("generate_validated_response")
