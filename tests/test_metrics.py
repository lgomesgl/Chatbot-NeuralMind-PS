import unittest
import numpy as np
from metrics.cossine_similarity import calculate_cosine_similarity, get_top_k_similar_documents
from metrics.generation import rouge_score, bleu_score

class TestCalculateCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity(self):
        query_embedding = np.array([[1, 0, 0]])
        document_embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        similarities = calculate_cosine_similarity(query_embedding, document_embeddings)
        
        expected = np.array([[1.0, 0.0, 0.0]])  # Expected similarity
        np.testing.assert_almost_equal(similarities, expected, decimal=5)

class TestGetTopKSimilarDocuments(unittest.TestCase):
    def test_top_k_similar_documents(self):
        query_embedding = np.array([[1, 0, 0]])
        document_embeddings = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
        indices, similarities = get_top_k_similar_documents(query_embedding, document_embeddings, k=2)

        self.assertListEqual(indices.tolist(), [0, 2])  # Top-2 most similar
        np.testing.assert_almost_equal(similarities, [1.0, 0.7071], decimal=4)
        
class TestRougeScore(unittest.TestCase):
    def test_rouge_score(self):
        generated_response = "The quick brown fox jumps over the lazy dog."
        reference_response = "The quick brown fox jumped over the lazy dog."
        
        result = rouge_score(generated_response, reference_response)
        self.assertIn("rouge-1", result)
        self.assertIn("rouge-2", result)
        self.assertIn("rouge-l", result)

class TestBleuScore(unittest.TestCase):
    def test_bleu_score(self):
        generated_response = "The quick brown fox jumps over the lazy dog."
        reference_response = "The quick brown fox jumped over the lazy dog."
        
        result = bleu_score(generated_response, reference_response)
        self.assertGreater(result, 0.5)
        
if __name__ == "__main__":
    unittest.main()
