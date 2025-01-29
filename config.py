import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMERGENCEAI_API_KEY = os.getenv("EMERGENCEAI_API_KEY")

# Model configurations
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's text embedding model
GCN_HIDDEN_CHANNELS = 64
GCN_NUM_LAYERS = 3

# Agent configurations
MAX_DISCUSSION_ROUNDS = 3
CONFIDENCE_THRESHOLD = 0.8  # Threshold for immediate consensus
VOTING_THRESHOLD = 0.5  # Threshold for majority decision
