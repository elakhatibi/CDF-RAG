import pinecone
import networkx as nx
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import TypedDict
from config import CONFIG

class RetrievalState(TypedDict):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]

class DocumentRetriever:
    def __init__(self):
        self.index_name = CONFIG["PINECONE_INDEX"]
        pinecone.init(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENV"])
        self.index = pinecone.Index(self.index_name)
        self.encoder = SentenceTransformer(CONFIG["ENCODER_MODEL"])
    
    def retrieve(self, query: str, top_k: int = 5):
        query_vector = self.encoder.encode(query).tolist()
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [item["metadata"]["text"] for item in results["matches"]]

class CausalGraphRetriever:
    def __init__(self):
        self.graph = nx.read_gpickle(CONFIG["CAUSAL_GRAPH_PATH"])
    
    def retrieve(self, query: str):
        related_nodes = []
        for node in self.graph.nodes:
            if query.lower() in node.lower():
                related_nodes.append(node)
                related_nodes.extend(nx.descendants(self.graph, node))
        return list(set(related_nodes))

class AdaptiveRetrieverAgent:
    def __init__(self):
        self.doc_retriever = DocumentRetriever()
        self.causal_retriever = CausalGraphRetriever()
    
    def retrieve_knowledge(self, state: RetrievalState) -> RetrievalState:
        query = state['query']
        state['retrieved_docs'] = self.doc_retriever.retrieve(query)
        state['causal_docs'] = self.causal_retriever.retrieve(query)
        return state

def build_retrieval_graph():
    builder = StateGraph(RetrievalState)
    builder.add_node("retrieve_knowledge", AdaptiveRetrieverAgent().retrieve_knowledge)
    builder.set_entry_point("retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", END)
    return builder.compile()
