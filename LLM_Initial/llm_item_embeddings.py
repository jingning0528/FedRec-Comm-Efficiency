import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_item_embeddings(item_texts, target_dim, model_name='all-MiniLM-L6-v2', save_path=None):
    """
    Generate item embeddings from text descriptions using a pretrained LLM.
    
    Args:
        item_texts: list of str, e.g. ["Toy Story (1995) | Animation, Children's, Comedy", ...]
        target_dim: int, the embedding dim used in NCF (2*predictive_factor)
        model_name: pretrained sentence transformer model
        save_path: optional path to save embeddings
    
    Returns:
        torch.Tensor of shape (num_items, target_dim)
    """
    model = SentenceTransformer(model_name)
    raw_embeddings = model.encode(item_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Project from LLM dim to target_dim using a linear projection
    llm_dim = raw_embeddings.shape[1]
    if llm_dim != target_dim:
        # Use PCA or a random projection to match target_dim
        from sklearn.decomposition import PCA
        if llm_dim > target_dim:
            pca = PCA(n_components=target_dim)
            raw_embeddings = pca.fit_transform(raw_embeddings)
        else:
            # Pad with zeros if LLM dim < target_dim
            padding = np.zeros((raw_embeddings.shape[0], target_dim - llm_dim))
            raw_embeddings = np.concatenate([raw_embeddings, padding], axis=1)
    
    # Normalize to have similar scale as random init (std≈0.01)
    raw_embeddings = raw_embeddings / np.std(raw_embeddings) * 0.01
    
    embeddings = torch.tensor(raw_embeddings, dtype=torch.float32)
    
    if save_path:
        torch.save(embeddings, save_path)
    
    return embeddings


def load_ml1m_item_texts(movies_path='./ml-1m/movies.dat'):
    """Load movie titles + genres from ML-1M as text descriptions."""
    item_texts = {}
    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2]
            item_texts[movie_id] = f"{title} | {genres}"
    return item_texts


def generate_item_embeddings_for_matrix(item_texts_dict, item_ids, target_dim, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings only for items that appear in the ratings matrix,
    in the correct order.
    
    Args:
        item_texts_dict: dict {movie_id: text}
        item_ids: list of movie IDs in the order they appear in the ratings matrix
        target_dim: embedding dimension (2*predictive_factor)
    """
    # Build text list in matrix column order
    texts = []
    for mid in item_ids:
        if mid in item_texts_dict:
            texts.append(item_texts_dict[mid])
        else:
            texts.append("unknown movie")  # fallback for missing
    
    return generate_item_embeddings(texts, target_dim, model_name)