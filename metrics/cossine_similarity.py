import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(query_embedding: np.ndarray, document_embeddings: np.ndarray):
    """
    Calcula a similaridade de cossenos entre a consulta e os documentos.

    Args:
        query_embedding (np.ndarray): O vetor de embeddings da consulta do usuário.
        document_embeddings (np.ndarray): Os embeddings dos documentos no índice.

    Returns:
        np.ndarray: A similaridade de cossenos entre a consulta e os documentos.
    """
    return cosine_similarity(query_embedding, document_embeddings)

def get_top_k_similar_documents(query_embedding: np.ndarray, document_embeddings: np.ndarray, k: int = 10):
    """
    Retorna os índices dos k documentos mais similares com base na similaridade de cossenos.

    Args:
        query_embedding (np.ndarray): O vetor de embeddings da consulta do usuário.
        document_embeddings (np.ndarray): Os embeddings dos documentos no índice.
        k (int): O número de documentos a retornar.

    Returns:
        list: Índices dos k documentos mais similares.
    """
    similarities = calculate_cosine_similarity(query_embedding, document_embeddings)
    sorted_indices = np.argsort(similarities[0])[::-1]  # Ordenar em ordem decrescente
    return sorted_indices[:k], similarities[0][sorted_indices[:k]]