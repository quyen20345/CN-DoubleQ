import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Initialize the Qdrant client as a singleton instance
# Default values are provided for robustness
QDRANT_CLIENT = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
    timeout=int(os.getenv("QDRANT_TIMEOUT", 20))
)

def get_qdrant_client() -> QdrantClient:
    """Returns the singleton Qdrant client instance."""
    return QDRANT_CLIENT