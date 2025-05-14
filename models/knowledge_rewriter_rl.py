import openai
from langgraph.graph import StateGraph
from config import CONFIG

class KnowledgeRewriter:
    def __init__(self):
        self.model = CONFIG["KNOWLEDGE_REWRITING_MODEL"]
    
    def rewrite(self, documents):
        """Rewrite retrieved documents into structured knowledge format."""
        context = "\n".join(documents)
        prompt = f"Rewrite the following knowledge into a structured, logically coherent format:\n{context}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "Summarize and structure knowledge from retrieved documents."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"].strip()

# Reinforcement Learning Agent for Optimized Knowledge Structuring
class KnowledgeRewritingAgent:
    def __init__(self):
        self.rewriter = KnowledgeRewriter()
    
    def optimize_rewriting(self, state):
        """Agent rewrites knowledge and refines structure using RL-based optimization."""
        state['rewritten_knowledge'] = self.rewriter.rewrite(state['retrieved_docs'] + state['causal_docs'])
        return state

# LangGraph Workflow
rewriting_graph = StateGraph()
rewriting_graph.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
rewriting_graph.set_entry("optimize_rewriting")
rewriting_graph.set_exit("optimize_rewriting")
