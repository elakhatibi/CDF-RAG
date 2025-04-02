import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

CONFIG = {
    "NEO4J_URI": "bolt://localhost:7687",  # Change if using a remote Neo4j server
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "elahekhatibi",  # Replace with your actual password

    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # Uses .env for security
    
    "KNOWLEDGE_REWRITING_MODEL": "gpt-4",

    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),  # If using Pinecone for vector retrieval

    "QUERY_REFINEMENT_MODEL": "gpt-4",  # Default OpenAI model
    "LLM_MODEL": "gpt-4",  # Model used for response generation
    "HALLUCINATION_DETECTION_MODEL": "gpt-4",  # Model for fact-checking

    "ENCODER_MODEL": "all-MiniLM-L6-v2",  # Sentence Transformer model for encoding
    "CAUSAL_GRAPH_PATH": "causal_graph.pkl",  # If storing the graph locally
}
