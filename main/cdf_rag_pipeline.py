import argparse
from langgraph.graph import StateGraph
from retrieval.query_refiner_rl import QueryRefinementAgent
from retrieval.document_causal_retriever_rl import AdaptiveRetrieverAgent
from models.knowledge_rewriter_rl import KnowledgeRewritingAgent
from models.llm_inference_rl import LLMResponseAgent
from models.hallucination_detector_rl import HallucinationCorrectionAgent

def main(query: str):
    print("Initializing CDF-RAG pipeline with LangGraph and RL")
    
    # Define LangGraph workflow
    graph = StateGraph()
    graph.add_node("refine_query", QueryRefinementAgent().optimize_query)
    graph.add_node("retrieve_knowledge", AdaptiveRetrieverAgent().retrieve_knowledge)
    graph.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
    graph.add_node("generate_validated_response", LLMResponseAgent().generate_validated_response)
    graph.add_node("correct_hallucinations", HallucinationCorrectionAgent().correct_hallucinations)
    
    # Define execution order
    graph.add_edge("refine_query", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "optimize_rewriting")
    graph.add_edge("optimize_rewriting", "generate_validated_response")
    graph.add_edge("generate_validated_response", "correct_hallucinations")
    
    graph.set_entry("refine_query")
    graph.set_exit("correct_hallucinations")
    
    state = {'query': query}
    final_state = graph.run(state)
    
    print("Final Response:", final_state['response'])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CDF-RAG pipeline with LangGraph and RL")
    parser.add_argument("query", type=str, help="User query")
    args = parser.parse_args()
    main(args.query)
