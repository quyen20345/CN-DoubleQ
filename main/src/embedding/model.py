import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv() 

class DenseEmbedding:
    def __init__(self, model_name=os.getenv("DENSE_MODEL")):
        self.model = SentenceTransformer(model_name, cache_folder=os.path.join(os.path.dirname(os.path.realpath(__file__)),"models"))

    def encode(self, texts):
        if isinstance(texts, str):
            return self.model.encode(texts)
        elif isinstance(texts, list):
            return [e.tolist() for e in self.model.encode(texts)]
        else:
            raise ValueError("Input must be a string or a list of strings.")
        
    def get_dimension(self):
        return self.encode("test").shape[0]