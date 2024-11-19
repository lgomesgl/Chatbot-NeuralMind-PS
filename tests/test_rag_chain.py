import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from langchain.schema import Document
from scr.rag_chain import (
    init_groq_client,
    split_documents_into_chunks,
    generate_embeddings,
    check_index_if_exist,
    load_index,
    create_or_load_faiss_index,
    nlu_process,
    dialog_manager,
    nlg_process,
    ml_block,
)

class TestInitGroqClient(unittest.TestCase):
    @patch('os.environ.get')
    @patch('scr.rag_chain.Groq')
    def test_init_groq_client(self, mock_groq, mock_env_get):
        mock_env_get.return_value = "fake_api_key"
        mock_client = MagicMock()
        mock_groq.return_value = mock_client

        client = init_groq_client()
        mock_env_get.assert_called_with("GROQ_API_KEY")
        mock_groq.assert_called_once_with(api_key="fake_api_key")
        self.assertEqual(client, mock_client)


class TestSplitDocumentsIntoChunks(unittest.TestCase):
    def test_split_documents_into_chunks(self):
        docs = [Document(page_content="This is a test document.")]
        chunks = split_documents_into_chunks(docs, chunk_size=10, chunk_overlap=5)
        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], Document)


class TestGenerateEmbeddings(unittest.TestCase):
    @patch('scr.rag_chain.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(5, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        texts = ["test1", "test2", "test3"]
        embeddings, model = generate_embeddings("test_model", texts)

        mock_sentence_transformer.assert_called_with("test_model")
        mock_model.encode.assert_called_with(texts, convert_to_tensor=True)
        self.assertTrue((embeddings == mock_embeddings).all())

class TestCheckIndexIfExist(unittest.TestCase):
    def test_check_index_if_exist(self):
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            self.assertTrue(check_index_if_exist("fake_path"))
            mock_exists.assert_called_with("fake_path")

class TestLoadIndex(unittest.TestCase):
    @patch('scr.rag_chain.faiss')
    def test_load_index(self, mock_faiss):
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index

        index = load_index("fake_path")
        mock_faiss.read_index.assert_called_with("fake_path")
        self.assertEqual(index, mock_index)

class TestCreateOrLoadFaissIndex(unittest.TestCase):
    @patch('scr.rag_chain.faiss')
    def test_create_or_load_faiss_index(self, mock_faiss):
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        embeddings = np.random.rand(10, 384)
        index = create_or_load_faiss_index(embeddings, "fake_path")

        mock_faiss.IndexFlatL2.assert_called_with(384)
        mock_index.add.assert_called_with(embeddings)
        mock_faiss.write_index.assert_called_with(mock_index, "fake_path")
        self.assertEqual(index, mock_index)

class TestNLUProcess(unittest.TestCase):
    @patch('scr.rag_chain.SentenceTransformer')
    def test_nlu_process(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_embedding = np.random.rand(1, 384)
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        query_embedding = nlu_process("user query", mock_model)
        mock_model.encode.assert_called_with(["user query"], convert_to_tensor=True)
        self.assertTrue((query_embedding == mock_embedding).all())

class TestDialogManager(unittest.TestCase):
    @patch('scr.rag_chain.get_top_k_similar_documents')
    def test_dialog_manager(self, mock_get_top_k_similar_documents):
        embeddings = np.random.rand(10, 384)
        chunk_texts = [f"Chunk {i}" for i in range(10)]
        query_embedding = np.random.rand(1, 384)

        mock_get_top_k_similar_documents.return_value = ([0, 1, 2], [0.9, 0.8, 0.7])
        index = MagicMock()
        index.search.return_value = (None, np.array([[0, 1, 2]]))

        context_rag, context_cossine = dialog_manager(query_embedding, index, chunk_texts, embeddings, k=3)

        self.assertIn("Chunk 0", context_rag)
        self.assertIn("Chunk 1", context_cossine)

class TestNLGProcess(unittest.TestCase):
    @patch('scr.rag_chain.init_groq_client')
    def test_nlg_process(self, mock_init_client):
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated response"))]
        mock_client.chat.completions.create.return_value = mock_response

        response = nlg_process("user query", "context", mock_client)
        self.assertEqual(response, "Generated response")

class TestMLBlock(unittest.TestCase):
    @patch('scr.rag_chain.init_groq_client')
    def test_ml_block(self, mock_init_client):
        # Mock dependencies
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client

        # Mock model
        model = MagicMock()
        mock_embedding = np.random.rand(1, 384)  # Simula um embedding válido
        model.encode.return_value = mock_embedding  # O modelo deve retornar um embedding válido

        # Mock FAISS index
        embeddings = np.random.rand(10, 384)  # Simula 10 embeddings com dimensão 384
        chunk_texts = [f"Chunk {i}" for i in range(10)]  # Simula 10 pedaços de texto
        index = MagicMock()
        index.search.return_value = (None, np.array([[0, 1, 2]]))  # Resposta mockada do FAISS

        # Mock OpenAI client response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Generated answer"))]
        )

        # Teste a função ml_block
        user_query = "Test query"
        answer, context_cossine = ml_block(user_query, model, index, chunk_texts, embeddings, mock_client)

        # Assertions
        self.assertIn("Generated answer", answer)  # Verifica se a resposta está correta
        self.assertIsNotNone(context_cossine)  # Garante que o contexto não é None
        self.assertGreater(len(context_cossine), 0)  # Garante que o contexto tem conteúdo

if __name__ == "__main__":
    unittest.main()
