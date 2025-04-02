import gym
import numpy as np
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers.util import cos_sim

# Load sentence encoder for semantic queries
encoder = SentenceTransformer("all-MiniLM-L6-v2")


# Load your fine-tuned LLM (e.g., LLaMA, Mixtral, etc.)
model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"  # or your LoRA checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 1. Simulate retrieval (can be real with FAISS/Chroma)
def retrieve_causal_docs(query):
    return [f"Retrieved doc for: {query}"], ["cause → effect"]

# 2. Use LLM to rewrite causal explanation
def call_rewriting_llm(docs, paths):
    prompt = f"Given the documents:\n{docs}\nand causal paths:\n{paths},\nWrite a structured causal explanation."
    response = llm_pipeline(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    return response.strip()

# 3. Use LLM to generate final response
def call_final_response_llm(query, explanation):
    prompt = f"Query: {query}\nExplanation: {explanation}\nNow provide a final natural language response."
    response = llm_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    return response.strip()

# 4. Hallucination detection (simple check)
def detect_hallucination(rewritten, final_answer):
    return rewritten not in final_answer  # placeholder logic, can be replaced with LLM judgment or RAG match

# 5. Average depth (mock, replace with real graph depth if needed)
def average_depth(paths):
    return np.mean([p.count("→") + 1 for p in paths]) if paths else 0

# 6. Cosine similarity between query and rewritten explanation
def cosine_sim(query, rewritten):
    emb1 = encoder.encode(query, normalize_embeddings=True)
    emb2 = encoder.encode(rewritten, normalize_embeddings=True)
    return cos_sim(emb1, emb2).item()

# Simulated retrieval and hallucination scoring
def simulate_cdf_pipeline(refined_query):
    # 1. Call retriever with refined_query
    retrieved_docs, causal_paths = retrieve_causal_docs(refined_query)

    # 2. Rewrite into structured causal explanation
    rewritten = call_rewriting_llm(retrieved_docs, causal_paths)

    # 3. Generate final response
    final_answer = call_final_response_llm(refined_query, rewritten)

    # 4. Run hallucination detection
    hallucinated = detect_hallucination(rewritten, final_answer)

    # 5. Compute metrics
    retrieval_coverage = 1 if retrieved_docs else 0
    depth = average_depth(causal_paths)
    context_relevance = cosine_sim(refined_query, rewritten)
    halluc_score = 1 - int(hallucinated)

    reward = (
        0.3 * retrieval_coverage +
        0.3 * depth +
        0.2 * context_relevance +
        0.2 * halluc_score
    )
    return reward, hallucinated

# ----------------------------
# RL Environment
# ----------------------------
class QueryRefinementEnv(gym.Env):
    def __init__(self, queries):
        super().__init__()
        self.queries = queries
        self.query_index = 0
        self.embedding_dim = 384

        # Actions: 0 = Expand, 1 = Simplify, 2 = Decompose
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.embedding_dim,), dtype=np.float32)

    def reset(self):
        self.query_index = (self.query_index + 1) % len(self.queries)
        self.current_query = self.queries[self.query_index]
        emb = encoder.encode(self.current_query, normalize_embeddings=True)
        return emb.astype(np.float32)

    def step(self, action):
        query = self.current_query
        strategy = ["Expand", "Simplify", "Decompose"][action]

        # Apply simulated strategy
        if strategy == "Expand":
            refined = f"What are the detailed causes of {query}?"
        elif strategy == "Simplify":
            refined = f"What causes {query.split()[-1]}?"
        else:  # Decompose
            refined = f"What are the subcauses of {query}?"

        reward, hallucinated = simulate_cdf_pipeline(refined)
        obs = encoder.encode(query, normalize_embeddings=True).astype(np.float32)
        done = False
        return obs, reward, done, {}

# ----------------------------
# Training Setup
# ----------------------------

if __name__ == "__main__":
    queries = [
        "Why do people quit jobs?",
        "Why is climate change a concern?",
        "What leads to student underperformance?",
        "Why do patients miss appointments?",
        "Why do people develop diabetes?",
        "Why is homelessness increasing?"
    ]

    env = QueryRefinementEnv(queries)
    model = PPO("MlpPolicy", env, verbose=1)

    # ✅ Set number of epochs
    total_epochs = 5
    steps_per_epoch = len(queries) * 100  # Total steps = epochs × queries × samples

    model.learn(total_timesteps=total_epochs * steps_per_epoch)
    model.save("ppo_query_refiner_2")
