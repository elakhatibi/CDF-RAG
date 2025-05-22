import openai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from config import CONFIG

class LLMState(TypedDict):
    query: str
    rewritten_knowledge: str
    response: str

class LLMInference:
    def __init__(self):
        self.model = CONFIG["LLM_MODEL"]
    
    def generate(self, context, query, force_retrieval=False):
        prompt = f"Using the following knowledge, answer the query:\n{context}\n\nQuery: {query}"
        if force_retrieval:
            prompt = "Revised Retrieval Required: " + prompt
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Generate factually aligned responses."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()

class LLMResponseAgent:
    def __init__(self):
        self.llm = LLMInference()
    
    def generate_validated_response(self, state: LLMState) -> LLMState:
        state['response'] = self.llm.generate(state['rewritten_knowledge'], state['query'])
        return state

def build_llm_graph():
    builder = StateGraph(LLMState)
    builder.add_node("generate_validated_response", LLMResponseAgent().generate_validated_response)
    builder.set_entry_point("generate_validated_response")
    builder.add_edge("generate_validated_response", END)
    return builder.compile()
