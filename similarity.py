from sentence_transformers import SentenceTransformer, util
import pickle
# embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')

with open("ingredient_embedding.pkl","rb") as f:
    ingredient_embedding = pickle.load(f)

def get_cosine(query_embeddings):
    distances = util.pytorch_cos_sim(query_embeddings, ingredient_embedding)[0]
    distances = distances.cpu()
    score = distances.sum().item()
    return score

