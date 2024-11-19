import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from metrics.cossine_similarity import calculate_cosine_similarity, get_top_k_similar_documents
from metrics.recovery import precision, recall, f1_score, mean_reciprocal_rank
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

class TestPrecision(unittest.TestCase):
    def test_precision(self):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        relevant_docs = ["doc1", "doc3", "doc5"]
        
        precision_at_2 = precision(retrieved_docs, relevant_docs, k=2)
        self.assertAlmostEqual(precision_at_2, 0.5)  # 1 relevant doc in top-2
        
        precision_at_4 = precision(retrieved_docs, relevant_docs, k=4)
        self.assertAlmostEqual(precision_at_4, 0.5)  # 2 relevant docs in top-4

class TestRecall(unittest.TestCase):
    def test_recall(self):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        relevant_docs = ["doc1", "doc3", "doc5"]
        
        recall_at_2 = recall(retrieved_docs, relevant_docs, k=2)
        self.assertAlmostEqual(recall_at_2, 1/3)  # 1 relevant doc out of 3
        
        recall_at_4 = recall(retrieved_docs, relevant_docs, k=4)
        self.assertAlmostEqual(recall_at_4, 2/3)  # 2 relevant docs out of 3

class TestF1Score(unittest.TestCase):
    def test_f1_score(self):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        relevant_docs = ["doc2", "doc4", "doc5"]
        
        result = f1_score(retrieved_docs, relevant_docs, k=3)
        expected_precision = 1 / 3
        expected_recall = 1 / 3
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        
        self.assertAlmostEqual(result, expected_f1, places=2)

class TestMeanReciprocalRank(unittest.TestCase):
    def test_mean_reciprocal_rank(self):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        relevant_docs = ["doc2", "doc4", "doc5"]
        
        result = mean_reciprocal_rank(retrieved_docs, relevant_docs)
        self.assertAlmostEqual(result, 1 / 2, places=2)

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
