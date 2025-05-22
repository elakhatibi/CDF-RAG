import openai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import CONFIG

class RewritingState(TypedDict):
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str

class KnowledgeRewriter:
    def __init__(self):
        self.model = CONFIG["KNOWLEDGE_REWRITING_MODEL"]
    
    def rewrite(self, documents):
        context = "\n".join(documents)
        prompt = f"Rewrite the following knowledge into a structured, logically coherent format:\n{context}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize and structure knowledge from retrieved documents."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()

class KnowledgeRewritingAgent:
    def __init__(self):
        self.rewriter = KnowledgeRewriter()
    
    def optimize_rewriting(self, state: RewritingState) -> RewritingState:
        state['rewritten_knowledge'] = self.rewriter.rewrite(state['retrieved_docs'] + state['causal_docs'])
        return state

def build_rewriting_graph():
    builder = StateGraph(RewritingState)
    builder.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
    builder.set_entry_point("optimize_rewriting")
    builder.add_edge("optimize_rewriting", END)
    return builder.compile()
