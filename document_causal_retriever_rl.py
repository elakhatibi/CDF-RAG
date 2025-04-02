from query_refiner_rl import QueryRefinementEnv
from knowledge_rewriter_rl import RewritingState, KnowledgeRewritingAgent
from neo4j import GraphDatabase
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph
from pydantic import BaseModel
import numpy as np
import os
from dotenv import load_dotenv
from llm_inference_response_generator import LLMResponseGenerator
from hallucination_detector_rl import HallucinationDetector
from openai import OpenAI
from gpt_causal_enricher import CausalEnricher  # Ensure this class exists

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load env vars (for OpenAI key, if needed)
load_dotenv()

# Fine-tuning

class QueryRefinementAgent:
    def __init__(self):
        self.model = os.getenv("OPENAI_FINE_TUNED_MODEL")  # e.g. "ft:gpt-3.5-turbo:your-org:refined:xyz123"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def refine(self, state):
        query = state.query

        prompt = f"""Refine the following query for causal discovery. Make it more specific, semantically rich, and grounded in domain-specific terminology.

Query: {query}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a query refinement assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        refined_query = response.choices[0].message.content.strip()
        state.refined_query_text = refined_query
        return state


# Add Hallucination Detection Agent
class HallucinationDetectionAgent:
    def __init__(self):
        self.detector = HallucinationDetector()

    def detect(self, state):
        is_bad = self.detector.detect(state.final_response, state.rewritten_knowledge)
        state.is_hallucination = is_bad
        print(f"🛡️ Hallucination Detected: {is_bad}")
        return state

# Add a Fallback Regenerator Agent
    
class HallucinationFallbackAgent:
    def __init__(self):
        self.generator = LLMResponseGenerator()

    def regenerate(self, state):
        if state.is_hallucination:
            print("🔁 Regenerating response due to detected hallucination...")
            response = self.generator.generate(
                knowledge=state.rewritten_knowledge,
                original_query=state.query
            )
            state.final_response = response
        return state

    
# Add the Response Generator Node to LangGraph
class ResponseGenerationAgent:
    def __init__(self):
        self.generator = LLMResponseGenerator()

    def generate_response(self, state):
        response = self.generator.generate(
            knowledge=state.rewritten_knowledge,
            original_query=state.query
        )
        state.final_response = response
        return state

# --------------------------------------
# 1. Neo4j Graph Wrapper
# --------------------------------------
class CausalGraphRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def retrieve_direct_causes(self, effect):
        query = "MATCH (c:Concept)-[:CAUSES]->(e:Concept {name: $effect}) RETURN c.name AS cause"
        with self.driver.session() as session:
            result = session.run(query, effect=effect)
            return [record["cause"] for record in result]

    def retrieve_multi_hop_causes(self, effect):
        query = """
        MATCH path = (c:Concept)-[:CAUSES*]->(e:Concept {name: $effect})
        RETURN [node IN nodes(path) | node.name] AS causal_path
        """
        with self.driver.session() as session:
            result = session.run(query, effect=effect)
            return [record["causal_path"] for record in result]

    def get_all_concepts(self):
        query = "MATCH (c:Concept) RETURN DISTINCT c.name AS concept"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["concept"] for record in result]
        
    def insert_causal_pair(self, cause: str, effect: str):
        query = """
        MERGE (c:Concept {name: $cause})
        MERGE (e:Concept {name: $effect})
        MERGE (c)-[:CAUSES]->(e)
        """
        with self.driver.session() as session:
            session.run(query, cause=cause, effect=effect)


# --------------------------------------
# 2. Query Refinement Agent
# --------------------------------------
# class QueryRefinementAgent:
#     def __init__(self):
#         self.model = PPO.load("ppo_query_refiner")
#         self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

#     def refine(self, state):
#         query = state.query
#         env = QueryRefinementEnv(causal_graph={})
#         obs, _ = env.reset()
#         action, _ = self.model.predict(obs, deterministic=True)
#         action_int = int(action)

#         action_map = {0: "Expand Query", 1: "Simplify Query", 2: "Decompose Query"}
#         refinement_type = action_map[action_int]
#         print("🧠 Suggested refinement:", refinement_type)

#         prompt = {
#             "Expand Query": f"What are broader or upstream causes related to '{query}'?",
#             "Simplify Query": f"What is a simpler way to describe '{query}'?",
#             "Decompose Query": f"What are sub-questions or components of '{query}'?"
#         }[refinement_type]

#         query_embedding = self.encoder.encode(prompt)
#         concepts = state.graph.get_all_concepts()
#         concept_embeddings = self.encoder.encode(concepts)
#         similarities = cosine_similarity([query_embedding], concept_embeddings)[0]
#         best_idx = int(np.argmax(similarities))
#         matched_concept = concepts[best_idx]

#         print(f"🔎 Matched concept: {matched_concept} (score={similarities[best_idx]:.2f})")
#         state.refined_query = matched_concept
#         return state

class QueryRefinementAgent:
    def __init__(self):
        self.model = PPO.load("ppo_query_refiner")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def refine(self, state):
        query = state.query
        env = QueryRefinementEnv(causal_graph={})
        obs, _ = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        action_int = int(action)

        action_map = {
            0: "Expand Query",
            1: "Simplify Query",
            2: "Decompose Query"
        }
        refinement_type = action_map[action_int]
        state.refinement_type = refinement_type
        print(f"🧠 Suggested refinement: {refinement_type}")

        # Generate a natural-language refined query using LLM
        prompt = f"""You are a smart assistant. The user asked: '{query}'. Your task is to {refinement_type.lower()} this query while keeping the domain intact. Provide a better refined query as a single sentence."""
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in generating domain-specific refined queries."},
                {"role": "user", "content": prompt}
            ]
        )
        refined_query = response.choices[0].message.content.strip()
        print(f"🔁 Refined Query (LLM): {refined_query}")

        # Use LLM-refined query for concept matching
        query_embedding = self.encoder.encode(refined_query)
        concepts = state.graph.get_all_concepts()
        concept_embeddings = self.encoder.encode(concepts)
        similarities = cosine_similarity([query_embedding], concept_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        matched_concept = concepts[best_idx]

        print(f"🔎 Matched concept: {matched_concept} (score={similarities[best_idx]:.2f})")
        state.refined_query = matched_concept
        state.refined_query_text = refined_query
        return state

# --------------------------------------
# 3. Retrieval Agent
# --------------------------------------
class RetrievalAgent:
    def __init__(self):
        self.enricher = CausalEnricher()

    def retrieve(self, state):
        query = state.refined_query
        graph = state.graph

        direct = graph.retrieve_direct_causes(query)
        multi = graph.retrieve_multi_hop_causes(query)

        if not direct and not multi:
            print(f"⚠️ No results in Neo4j for: {query}. GPT will generate causal pairs...")
            new_pairs = self.enricher.generate_pairs(query)

            for cause, effect in new_pairs:
                graph.insert_causal_pair(cause.strip(), effect.strip())
            
            print(f"🔁 GPT enriched Neo4j with {len(new_pairs)} new causal pairs.")


            # Try retrieval again after enrichment
            direct = graph.retrieve_direct_causes(query)
            multi = graph.retrieve_multi_hop_causes(query)

        state.retrieved_docs = direct
        state.causal_docs = [f"{' → '.join(path)}" for path in multi]
        return state
# --------------------------------------
# 4. Rewriting Agent (GPT)
# --------------------------------------
class RewritingAgent:
    def __init__(self):
        self.rewriter = KnowledgeRewritingAgent().optimize_rewriting

    def rewrite(self, state):
        rewriting_state = RewritingState(
            query=state.query,
            retrieved_docs=state.retrieved_docs,
            causal_docs=state.causal_docs
        )

        result = self.rewriter(rewriting_state)
        state.rewritten_knowledge = result.rewritten_knowledge
        return state

# --------------------------------------
# 5. LangGraph State Schema
# --------------------------------------
class CDFState(BaseModel):
    query: str
    refinement_type: str = ""           # Type: Expand/Simplify/Decompose
    refined_query_text: str = ""        # LLM-generated refined NL query
    refined_query: str = ""             # Matched concept in graph
    retrieved_docs: list[str] = []      # Direct causes from Neo4j
    causal_docs: list[str] = []         # Multi-hop causes from Neo4j
    rewritten_knowledge: str = ""       # GPT structured explanation
    final_response: str = ""            # Final user-facing answer
    is_hallucination: bool = False      # True if hallucination detected
    graph: object                      # Reference to Neo4j connector

    class Config:
        arbitrary_types_allowed = True

# --------------------------------------
# 6. Define LangGraph Pipeline
# --------------------------------------
print("⚙️ Initializing LangGraph pipeline...")
pipeline = StateGraph(CDFState)

pipeline.add_node("refine_query", QueryRefinementAgent().refine)
pipeline.add_node("retrieve_knowledge", RetrievalAgent().retrieve)
pipeline.add_node("rewrite_knowledge", RewritingAgent().rewrite)
pipeline.add_node("generate_response", ResponseGenerationAgent().generate_response)
pipeline.add_node("detect_hallucination", HallucinationDetectionAgent().detect)
pipeline.add_node("regenerate_response_if_needed", HallucinationFallbackAgent().regenerate)


graph = (
    pipeline
    .set_entry_point("refine_query")
    .add_edge("refine_query", "retrieve_knowledge")
    .add_edge("retrieve_knowledge", "rewrite_knowledge")
    .add_edge("rewrite_knowledge", "generate_response")  # ✅ new edge
    .add_edge("generate_response", "detect_hallucination")
    .add_edge("detect_hallucination", "regenerate_response_if_needed")
    .set_finish_point("regenerate_response_if_needed")
    .compile()
)

print("✅ LangGraph pipeline compiled.")

# --------------------------------------
# 7. Run Full Pipeline
# --------------------------------------
if __name__ == "__main__":
    # Replace with your credentials or read from config
    graph_obj = CausalGraphRetriever("bolt://localhost:7687", "neo4j", "elahekhatibi")

    initial_state = {
        "query": "What causes heart disease?",
        "graph": graph_obj
    }

    final_state = graph.invoke(initial_state)

    print("\n🔹 FINAL STRUCTURED EXPLANATION:\n")
    print(final_state.get("rewritten_knowledge"))

    print("\n🧠 FINAL RESPONSE TO USER:\n")
    print(final_state.get("final_response"))
    print(f"\n🛡️ Is the response hallucinated? {'Yes' if final_state.get('is_hallucination') else 'No'}")



    graph_obj.close()
