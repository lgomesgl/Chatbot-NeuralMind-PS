import os
import sys
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../metrics')))
# from metrics.cossine_similarity import get_top_k_similar_documents

def init_groq_client():
    '''
        Initializes the Groq client with the provided API key
    '''
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def split_documents_into_chunks(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits documents into smaller chunks of text.

    Parameters:
    - docs (list): List of Document objects.
    - chunk_size (int): Maximum size of each chunk.
    - chunk_overlap (int): Overlap size between consecutive chunks.

    Returns:
    - list: A list of chunks (smaller text segments).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def generate_embeddings(model_name: str, texts: list):
    """
    Generates embeddings for a list of texts using SentenceTransformer.

    Parameters:
    - model_name (str): Name of the SentenceTransformer model.
    - texts (list): List of text strings.

    Returns:
    - Tuple[np.ndarray, SentenceTransformer]: Embeddings array and the SentenceTransformer model instance.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings, model

def create_index_faiss(embeddings: np.ndarray):
    """
    Creates a FAISS index with the provided embeddings or loads an existing index.

    Parameters:
    - embeddings (np.ndarray): Text embeddings to be indexed.

    Returns:
    - faiss.Index: The created FAISS index.
    """
    embedding_dim = embeddings.shape[1]

    # Creates a FAISS index based on Euclidean distance (L2)
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Adds the embeddings to the index
    index.add(embeddings)

    return index

def nlu_process(user_query: str, model: SentenceTransformer):
    """
    Encodes the user query into an embedding.

    Parameters:
    - user_query (str): User query string.
    - model (SentenceTransformer): SentenceTransformer model instance.

    Returns:
    - np.ndarray: Embedding of the user query.
    """
    return model.encode([user_query], convert_to_tensor=True)

def dialog_manager(query_embedding: np.ndarray, index: faiss.IndexFlatL2, chunk_texts: list, embeddings: np.ndarray, k: int = 10):
    """
    Retrieves relevant context based on the user's query using FAISS and cosine similarity.

    Parameters:
    - query_embedding (np.ndarray): Embedding of the user query.
    - index (faiss.IndexFlatL2): FAISS index instance.
    - chunk_texts (list): List of text chunks.
    - embeddings (np.ndarray): Array of embeddings for the text chunks.
    - k (int): Number of top results to retrieve.

    Returns:
    - Tuple[str, str]: Relevant context retrieved using FAISS and cosine similarity.
    """
    # Ensure embeddings are 2D
    query_embedding = np.atleast_2d(query_embedding)
    embeddings = np.atleast_2d(embeddings)

    # Check for empty arrays
    if query_embedding.size == 0 or embeddings.size == 0:
        raise ValueError("Query embedding or embeddings array is empty.")

    # Retrieve top-k chunks using FAISS
    _, indices = index.search(query_embedding, k)
    relevant_chunks_rag = [chunk_texts[i] for i in indices[0]]
    context_rag = "\n\n".join(relevant_chunks_rag)

    # # Retrieve top-k chunks using cosine similarity
    # top_k_indices, _ = get_top_k_similar_documents(query_embedding, embeddings, k)
    # relevant_chunks_cossine = [chunk_texts[i] for i in top_k_indices]
    # context_cossine = "\n\n".join(relevant_chunks_cossine)

    return context_rag

def nlg_process(user_query: str, context: str, client):
    """
    Generates a natural language response using a language model.

    Parameters:
    - user_query (str): User query string.
    - context (str): Relevant context for generating the response.
    - client: API client for the language model.

    Returns:
    - str: Generated response.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Você é um chatbot inteligente e deve fornecer uma resposta precisa e relevante à pergunta do usuário sobre o vestibular da Unicamp a partir do banco de informações relacionado a pergunta. Transforma o texto em um texto humanizado. Troque VU pelo Vestibular Unicamp. Aqui está seu banco de informçãoes:\n\n{context}"
            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model="llama3-70b-8192"
    )
    return response.choices[0].message.content


def ml_block(user_query: str, 
             model: SentenceTransformer, 
             index: faiss.IndexFlatL2, 
             chunk_texts: list, 
             embeddings: np.ndarray, 
             client):
    """
    Executes the entire NLP pipeline: NLU, Dialog Management, and NLG.

    Parameters:
    - user_query (str): User query string.
    - model (SentenceTransformer): SentenceTransformer model instance.
    - index (faiss.IndexFlatL2): FAISS index instance.
    - chunk_texts (list): List of text chunks.
    - embeddings (np.ndarray): Array of embeddings for the text chunks.
    - client: API client for the language model.

    Returns:
    - Tuple[str, str]: Generated answer and cosine similarity context.
    """
    try:
        # Step 1: Process the user query to generate embeddings
        query_embedding = nlu_process(user_query, model)

        # Step 2: Retrieve relevant context using FAISS and cosine similarity
        context_rag = dialog_manager(query_embedding, index, chunk_texts, embeddings)
        
        # Step 3: Generate a natural language response
        answer_rag = nlg_process(user_query, context_rag, client)

        return answer_rag
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}", ""    
    
