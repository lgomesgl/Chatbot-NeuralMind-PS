import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(query_embedding: np.ndarray, document_embeddings: np.ndarray):
    """
    Calculates the cosine similarity between the query and the documents.

    Args:
        query_embedding (np.ndarray): The embedding vector of the user query.
        document_embeddings (np.ndarray): The embeddings of the documents in the index.

    Returns:
        np.ndarray: The cosine similarity between the query and the documents.
    """
    return cosine_similarity(query_embedding, document_embeddings)

def get_top_k_similar_documents(query_embedding: np.ndarray, document_embeddings: np.ndarray, k: int = 10):
    """
    Returns the indices of the top-k most similar documents based on cosine similarity.

    Args:
        query_embedding (np.ndarray): The embedding vector of the user query.
        document_embeddings (np.ndarray): The embeddings of the documents in the index.
        k (int): The number of documents to return.

    Returns:
        list: Indices of the top-k most similar documents.
    """
    similarities = calculate_cosine_similarity(query_embedding, document_embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]  # Sort in descending order
    return sorted_indices[:k], similarities[0][sorted_indices[:k]]
