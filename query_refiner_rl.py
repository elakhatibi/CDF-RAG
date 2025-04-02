import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from config import CONFIG
from langgraph.graph import StateGraph
from pydantic import BaseModel

# ✅ Define Pydantic schema for LangGraph state
class QueryState(BaseModel):
    query: str
    refined_query: str

# ✅ Define Query Refinement Environment for RL (PPO)
class QueryRefinementEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for Query Refinement.
    The agent decides whether to Expand, Simplify, or Decompose a query.
    """

    def __init__(self, causal_graph):
        super(QueryRefinementEnv, self).__init__()

        # Store causal graph
        self.causal_graph = causal_graph

        # Define action space: [0: Expand, 1: Simplify, 2: Decompose]
        self.action_space = gym.spaces.Discrete(3)

        # Observation space = 1-dimensional float value (query complexity proxy)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Initial state (random complexity value)
        self.current_state = np.array([0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.array([np.random.rand()], dtype=np.float32)
        return self.current_state, {}

    def step(self, action):
        reward = np.random.randint(1, 10)
        next_state = np.array([np.random.rand()], dtype=np.float32)
        return next_state, reward, False, False, {}

# ✅ Define LangGraph Workflow using QueryState schema
print("Initializing LangGraph workflow...")
builder = StateGraph(QueryState)

# ✅ Define Query Refinement Node
def refine_query(state: QueryState) -> dict:
    return {"refined_query": f"Optimized: {state.query}"}

# ✅ Add node to the builder
builder.add_node("refine_query", refine_query)

# ✅ Define the full LangGraph flow (entry + exit)
graph = builder.set_entry_point("refine_query").set_finish_point("refine_query").compile()


print("LangGraph graph compiled successfully.")

# ✅ Gym environment validation (run standalone)
if __name__ == "__main__":
    env = QueryRefinementEnv(causal_graph={})
    check_env(env)
    print("✅ QueryRefinementEnv passed Gymnasium check!")


# def compute_reward(result):
#     retrieved_docs = result.get("retrieved_docs", [])
#     causal_chains = result.get("causal_docs", [])
#     hallucination = result.get("is_hallucination", False)

#     retrieval_score = len(retrieved_docs)
#     chain_depth_score = sum(len(path.split("→")) for path in causal_chains) / max(len(causal_chains), 1)
#     hallucination_penalty = 0 if hallucination else 1

#     return (
#         0.4 * retrieval_score +
#         0.3 * chain_depth_score +
#         0.3 * hallucination_penalty
#     )

